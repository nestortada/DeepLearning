"""Run local academic inference from a trained PAPILA checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from pickle import UnpicklingError

import torch
from PIL import Image

from .data import Normalization, make_image_transform
from .device import select_device
from .model import CLASSES, load_checkpoint


def checkpoint_transform(checkpoint: dict):
    return make_image_transform(
        checkpoint["architecture"]["image_size"], Normalization(**checkpoint["normalization"]),
        preserve_aspect=checkpoint["preprocessing"].get("geometry") == "letterbox",
    )


@torch.inference_mode()
def score_image(model, image: Image.Image, transform, device: torch.device | str) -> float:
    device = torch.device(device)
    pixels = transform(image if image.mode == "RGB" else image.convert("RGB")).unsqueeze(0).to(device)
    with torch.autocast(device.type, enabled=False):
        score = torch.sigmoid(model(pixels)).item()
    if not math.isfinite(score):
        raise ValueError("El checkpoint produjo un puntaje no finito.")
    return score


def predict_image(
    checkpoint_path: Path, image_path: Path, device: torch.device | str = "cpu", *, threshold: float | None = None,
) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    threshold = checkpoint["config"]["threshold"] if threshold is None else threshold
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar en [0, 1].")
    with Image.open(image_path) as image:
        score = score_image(model, image, checkpoint_transform(checkpoint), device)
    label = int(score >= threshold)
    return {"score_glaucoma": score, "threshold": threshold, "label": label,
            "class": CLASSES[label], "selected_epoch": checkpoint["epoch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, help="Umbral operativo fijado en validación.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = select_device(prefer_cuda=not args.cpu, enable_amp=False)
    try:
        result = predict_image(args.checkpoint, args.image, device.device, threshold=args.threshold)
    except (OSError, ValueError, RuntimeError, KeyError, UnpicklingError, Image.DecompressionBombError) as error:
        parser.error(str(error))
    print(json.dumps({**result, "device": str(device.device), "use": "puntaje académico sin calibrar"},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
