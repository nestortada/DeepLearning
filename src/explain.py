"""FP32 Grad-CAM for the positive logit of PAPILA transfer classifiers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import get_ident

import torch
from PIL import Image
from torch import nn
from torch.nn.functional import interpolate

from .data import Letterbox
from .predict import checkpoint_transform
from .pretrained import PapilaTransferModel


TARGET_LAYER = "backbone.layer4[-1]"


@dataclass(frozen=True)
class GradCAMResult:
    score_glaucoma: float
    label: int
    threshold: float
    heatmap: torch.Tensor  # Detached CPU tensor indexed by (y, x).
    content_box: tuple[int, int, int, int]  # Left, top, right, bottom (exclusive).
    feature_shape: tuple[int, int]


def letterbox_content_box(
    original_size: tuple[int, int], image_size: int,
) -> tuple[int, int, int, int]:
    """Get the exact paste box using the same Pillow transform as inference."""
    with Image.new("L", original_size, 255) as mask:
        box = Letterbox(image_size)(mask).getbbox()
    if box is None:
        raise ValueError("El letterbox no contiene píxeles de la imagen original.")
    return box


def _finite_map(heatmap: torch.Tensor) -> torch.Tensor:
    valid = heatmap.isfinite()
    minimum = heatmap.masked_fill(~valid, torch.inf).amin()
    fill = torch.where(minimum.isfinite(), minimum, 0.0)
    return torch.where(valid, heatmap, fill)


def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """Scale a nonnegative map to [0, 1], with zero for uniform or invalid maps."""
    values = _finite_map(heatmap).clamp_min(0)
    peak = values.amax()
    shifted = values - values.amin()
    span = shifted.amax()
    # Bilinear interpolation of a constant may introduce a few FP32 rounding steps.
    has_contrast = span > peak * (4 * torch.finfo(values.dtype).eps)
    return torch.where(has_contrast, shifted / torch.where(has_contrast, span, 1.0), 0.0)


def align_heatmap(
    feature_map: torch.Tensor, image_size: int, content_box: tuple[int, int, int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Interpolate to input space, crop the letterbox and project onto native pixels."""
    left, top, right, bottom = content_box
    if not (0 <= left < right <= image_size and 0 <= top < bottom <= image_size):
        raise ValueError("La caja de contenido no es válida para la entrada del modelo.")
    padded = interpolate(
        _finite_map(feature_map)[None, None], size=(image_size, image_size),
        mode="bilinear", align_corners=False,
    )
    content = normalize_heatmap(padded[0, 0, top:bottom, left:right])
    width, height = original_size
    aligned = interpolate(content[None, None], size=(height, width), mode="bilinear", align_corners=False)
    return aligned[0, 0].clamp(0, 1)


def explain_image(
    model: PapilaTransferModel, image: Image.Image, checkpoint: dict, *, threshold: float,
    output_size: tuple[int, int] | None = None,
) -> GradCAMResult:
    """Explain glaucoma evidence, including for images classified as negative.

    Pass a loaded FP32 model in eval mode and the chosen threshold explicitly.
    The detached CPU map defaults to the original size. Pass the dimensions
    of a full-image preview to project directly onto its display pixels.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Grad-CAM requiere ejecutar fuera de torch.inference_mode().")
    if any(module.training for module in model.modules()):
        raise ValueError("Todos los módulos deben estar en modo evaluación.")
    if checkpoint["preprocessing"].get("geometry") != "letterbox":
        raise ValueError("Grad-CAM requiere el preprocesamiento letterbox del checkpoint.")
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar en [0, 1].")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise ValueError("Grad-CAM requiere un modelo FP32.")

    image_size = checkpoint["architecture"]["image_size"]
    box = letterbox_content_box(image.size, image_size)
    transform = checkpoint_transform(checkpoint)
    pixels = transform(image if image.mode == "RGB" else image.convert("RGB")).unsqueeze(0).to(parameter.device)
    activation = None
    caller = get_ident()

    def capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        nonlocal activation
        if get_ident() == caller:
            activation = output

    with torch.enable_grad(), torch.autocast(parameter.device.type, enabled=False):
        # A frozen backbone needs a differentiable input to build the activation graph.
        pixels.requires_grad_(True)
        hook = model.gradcam_layer.register_forward_hook(capture)
        try:
            logits = model(pixels)
        finally:
            hook.remove()
        if activation is None or not activation.requires_grad:
            raise RuntimeError("La capa objetivo no produjo activaciones diferenciables.")
        if logits.shape != (1,) or logits.dtype != torch.float32 or not logits.isfinite().item():
            raise ValueError("Se esperaba un logit de glaucoma FP32 finito.")
        gradients, = torch.autograd.grad(logits[0], activation)
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        feature_map = (weights * activation.detach()).sum(dim=1)[0].relu()
        score = logits.detach().sigmoid().item()
        aligned = align_heatmap(feature_map, image_size, box, output_size if output_size is not None else image.size)

    return GradCAMResult(score, int(score >= threshold), threshold, aligned.cpu(), box, tuple(feature_map.shape))


def summarize_regions(heatmap: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> dict:
    """Measure map mass relative to area in aligned, caller-supplied boolean masks."""
    if heatmap.ndim != 2 or not heatmap.numel():
        raise ValueError("Se requiere un mapa bidimensional no vacío.")
    values = _finite_map(heatmap).clamp_min(0)
    total = values.sum(dtype=torch.float64).item()
    regions = {}
    for name, mask in masks.items():
        if mask.dtype != torch.bool or mask.shape != heatmap.shape:
            raise ValueError(f"Máscara booleana no alineada: {name}")
        mask = mask.to(values.device)
        area = mask.count_nonzero().item() / values.numel()
        mass = values[mask].sum(dtype=torch.float64).item() / total if total else None
        regions[name] = {
            "area_fraction": area, "cam_fraction": mass,
            "cam_to_area_ratio": mass / area if mass is not None and area else None,
        }
    return regions
