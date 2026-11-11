"""Train PAPILA classifiers using training and validation data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss

from .data import Sample, make_loaders
from .device import DeviceConfig, select_device
from .model import ARCHITECTURE, CHECKPOINT_VERSION, CLASSES, PapilaCNN, read_checkpoint
from .pretrained import (
    BACKBONES, EFFICIENTNET_B0, RESNET18, WEIGHT_DIGESTS, WEIGHTS_SHA256,
    PapilaResNet18, PapilaTransferModel, imagenet_normalization,
)


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 96
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 0.001
    epochs: int = 40
    patience: int = 5
    seed: int = 42
    weight_decay: float = 0.0
    dropout: float = 0.0
    augment_train: bool = True
    threshold: float = 0.5
    architecture: str = ARCHITECTURE
    tune_last_block: bool = False
    positive_weight: float = 1.0
    focal_loss: bool = False
    cache_preprocessed: bool = False
    tune_blocks: int = 0
    freeze_batch_norm: bool = True
    progressive_head_epochs: int = 0
    backbone_learning_rate: float = 0.0
    scheduler: str = "constant"
    augmentation: str = "flip"
    persistent_workers: bool = False
    optimizer: str = "adam"
    momentum: float = 0.9
    label_smoothing: float = 0.0
    accumulation_steps: int = 1

    @property
    def tuned_stages(self) -> int:
        return self.tune_blocks or int(self.tune_last_block)

    def __post_init__(self) -> None:
        if min(self.epochs, self.patience, self.batch_size, self.accumulation_steps) < 1 or self.image_size < 4 or self.num_workers < 0:
            raise ValueError("Épocas, paciencia y batch deben ser positivos, resolución ≥ 4 y workers ≥ 0.")
        if not (0 <= self.seed < 2**32 and 0 <= self.dropout <= 1 and 0 <= self.threshold <= 1):
            raise ValueError("Semilla fuera de rango o dropout/umbral fuera de [0, 1].")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0
                and math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError("Learning rate debe ser positivo y weight decay no negativo, ambos finitos.")
        if self.architecture not in (ARCHITECTURE, *BACKBONES):
            raise ValueError(f"Arquitectura desconocida: {self.architecture}")
        if self.tune_blocks not in (0, 1, 2) or (self.tune_last_block and self.tune_blocks):
            raise ValueError("Elige --tune-last-block o --tune-blocks con uno o dos bloques.")
        if self.tuned_stages and self.architecture not in BACKBONES:
            raise ValueError("El ajuste por bloques requiere un modelo preentrenado.")
        if not 0 <= self.progressive_head_epochs < self.epochs:
            raise ValueError("La fase de cabeza debe terminar antes de la última época.")
        if not math.isfinite(self.backbone_learning_rate) or self.backbone_learning_rate < 0:
            raise ValueError("La tasa del backbone debe ser finita y no negativa.")
        if not self.tuned_stages and (
            self.progressive_head_epochs or self.backbone_learning_rate or not self.freeze_batch_norm
        ):
            raise ValueError("Estas opciones requieren ajustar al menos un bloque del backbone.")
        if self.scheduler not in {"constant", "plateau", "cosine"}:
            raise ValueError("Planificador de tasa desconocido.")
        if self.augmentation not in {"flip", "photometric", "rotation", "brightness", "contrast", "zoomout"}:
            raise ValueError("Aumento desconocido.")
        if self.optimizer not in {"adam", "adamw", "sgd"} or not 0 <= self.momentum < 1:
            raise ValueError("Optimizador desconocido o momentum fuera de [0, 1).")
        if not 0 <= self.label_smoothing < 1 or (self.focal_loss and self.label_smoothing):
            raise ValueError("El suavizado en [0, 1) se admite con BCE.")
        if not math.isfinite(self.positive_weight) or self.positive_weight <= 0:
            raise ValueError("El peso de la clase positiva debe ser positivo y finito.")
        if self.focal_loss and self.positive_weight != 1.0:
            raise ValueError("Usa BCE ponderada o focal loss en corridas separadas.")


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = math.inf
    stale_epochs: int = 0

    def step(self, loss: float) -> tuple[bool, bool]:
        """Report strict improvement and whether consecutive stalls exhaust patience."""
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved, self.stale_epochs >= self.patience


def _seed_run(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _train_epoch(
    model: nn.Module, loader: DataLoader[Sample], device: DeviceConfig,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, accumulation_steps: int = 1,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    samples = 0
    skipped = 0
    updates = 0
    total_batches = len(loader)
    group_samples = 0
    if accumulation_steps > 1 and loader.batch_size is None:
        raise ValueError("La acumulación requiere un tamaño de lote fijo.")
    total_samples = len(loader.dataset)
    if accumulation_steps > 1 and loader.drop_last:
        total_samples = total_batches * loader.batch_size
    for batch_number, batch in enumerate(loader, start=1):
        images, labels = device.move_batch(batch)
        batch_samples = labels.numel()
        if (batch_number - 1) % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            if accumulation_steps > 1:
                group_samples = min(loader.batch_size * accumulation_steps, total_samples - samples)
        with device.autocast():
            loss = criterion(model(images), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Pérdida de entrenamiento no finita en lote {batch_number}: {value}.")
        backward_loss = loss if accumulation_steps == 1 else loss * (batch_samples / group_samples)
        scaler.scale(backward_loss).backward()
        if batch_number % accumulation_steps == 0 or batch_number == total_batches:
            if not scaler.is_enabled():
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or not torch.stack([gradient.isfinite().all() for gradient in gradients]).all().item():
                    raise FloatingPointError(f"Gradientes ausentes o no finitos en lote {batch_number}.")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            updates += 1
        loss_sum += value * batch_samples
        samples += batch_samples
    if skipped == updates:
        raise FloatingPointError("GradScaler omitió todas las actualizaciones de la época.")
    return loss_sum / samples, skipped


def classification_metrics(labels: torch.Tensor, predicted: torch.Tensor) -> dict:
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    return {
        "accuracy": (tn + tp) / labels.numel(),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "f2": 5 * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else 0.0,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def validation_metrics(labels: torch.Tensor, logits: torch.Tensor, threshold: float) -> dict:
    """Use glaucoma as positive and keep AP separate from trapezoidal PR-AUC."""
    metrics = classification_metrics(labels, torch.sigmoid(logits) >= threshold)
    metrics.update(roc_auc=None, average_precision=None, pr_auc=None)
    if metrics["tn"] + metrics["fp"] and metrics["tp"] + metrics["fn"]:
        precision, recall, _ = precision_recall_curve(labels, logits)
        metrics.update(roc_auc=float(roc_auc_score(labels, logits)),
                       average_precision=float(average_precision_score(labels, logits)),
                       pr_auc=float(auc(recall, precision)))
    return {f"val_{name}": value for name, value in metrics.items()}


@torch.inference_mode()
def validation_outputs(
    model: nn.Module, loader: DataLoader[Sample], device: DeviceConfig, criterion: nn.BCEWithLogitsLoss,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    scores, targets = [], []
    loss_sum = 0.0
    for batch_number, batch in enumerate(loader, start=1):
        images, labels = device.move_batch(batch)
        logits = model(images)  # Match FP32 inference for checkpoint comparison.
        loss = criterion(logits, labels).item()
        if not math.isfinite(loss):
            raise FloatingPointError(f"Pérdida de validación no finita en lote {batch_number}: {loss}.")
        loss_sum += loss * labels.numel()
        scores.append(logits.cpu())
        targets.append(labels.cpu())
    labels = torch.cat(targets)
    return loss_sum / labels.numel(), labels, torch.cat(scores)


def _validate(
    model: nn.Module, loader: DataLoader[Sample], device: DeviceConfig,
    criterion: nn.BCEWithLogitsLoss, threshold: float,
) -> dict:
    loss, labels, logits = validation_outputs(model, loader, device, criterion)
    return {"val_loss": loss, **validation_metrics(labels, logits, threshold)}


def _save_checkpoint(path: Path, payload: dict) -> None:
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(payload, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_loss(history: list[dict], best_epoch: int, output: Path, architecture: str, train_loss_name: str) -> None:
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_loss"] for row in history], marker=".", label=f"Train · {train_loss_name}")
    axis.plot(epochs, [row["val_loss"] for row in history], marker=".", label="Validación · BCE")
    axis.axvline(best_epoch, linestyle="--", color="gray", label=f"Época seleccionada ({best_epoch})")
    axis.set(title=f"PAPILA · {architecture} · pérdida por época", xlabel="Época", ylabel="Pérdida")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output, dpi=150)


def _plot_validation(history: list[dict], best_epoch: int, output: Path, threshold: float) -> None:
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    epochs = [row["epoch"] for row in history]
    for field, label in (
        ("recall", f"Sensibilidad a {threshold:g}"), ("specificity", f"Especificidad a {threshold:g}"),
        ("roc_auc", "ROC-AUC"), ("average_precision", "AP"),
    ):
        axis.plot(epochs, [row[f"val_{field}"] for row in history], marker=".", label=label)
    axis.axvline(best_epoch, linestyle="--", color="gray")
    axis.set(title="PAPILA · métricas de validación", xlabel="Época", ylim=(0, 1.02))
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output, dpi=150)


def run_training(
    dataset_dir: Path, manifest: Path, run_dir: Path, config: TrainConfig,
    device: DeviceConfig | None = None, *, init_checkpoint: Path | None = None,
    shared_cache: dict[str, dict[int, torch.Tensor]] | None = None,
) -> dict:
    """Fit on train, checkpoint on validation loss, and restore the selected weights."""
    if run_dir.exists():
        raise FileExistsError(f"La carpeta de la corrida ya existe: {run_dir}. Usa otra ruta.")
    if init_checkpoint is not None and not (
        config.architecture == RESNET18 and config.tuned_stages == 1 and config.dropout == 0
    ):
        raise ValueError("Un checkpoint inicial requiere ajustar un bloque de ResNet18 sin dropout.")
    _seed_run(config.seed)
    device = device if device is not None else select_device()
    use_cuda = device.device.type == "cuda"
    if use_cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device.device)
    started = time.perf_counter()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    transfer_learning = config.architecture in BACKBONES
    loaders = make_loaders(
        dataset_dir, manifest, device=device, batch_size=config.batch_size,
        num_workers=config.num_workers, image_size=config.image_size, seed=config.seed,
        augment_train=config.augment_train, include_test=False,
        normalization=imagenet_normalization(config.architecture) if transfer_learning else None,
        preserve_aspect=transfer_learning,
        cache_preprocessed=config.cache_preprocessed,
        augmentation=config.augmentation, persistent_workers=config.persistent_workers, shared_cache=shared_cache,
    )
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
        raise ValueError("El manifiesto cambió durante la carga.")

    if config.architecture == RESNET18:
        model = PapilaResNet18(pretrained=init_checkpoint is None, tune_blocks=config.tuned_stages,
                              freeze_batch_norm=config.freeze_batch_norm, dropout=config.dropout)
    elif transfer_learning:
        model = PapilaTransferModel(config.architecture, pretrained=True, tune_blocks=config.tuned_stages,
                                   freeze_batch_norm=config.freeze_batch_norm, dropout=config.dropout)
    else:
        model = PapilaCNN(config.image_size, config.dropout)
    initialization = None
    if init_checkpoint is not None:
        content = init_checkpoint.read_bytes()
        initial = read_checkpoint(io.BytesIO(content))
        source_weights = initial.get("pretrained_weights") or {}
        if (
            initial["architecture"] != {"name": RESNET18, "image_size": config.image_size, "dropout": 0.0}
            or initial.get("manifest_sha256") != digest
            or initial.get("normalization") != asdict(loaders.normalization)
            or initial.get("preprocessing", {}).get("geometry") != "letterbox"
            or source_weights.get("sha256") != WEIGHTS_SHA256
            or source_weights.get("backbone_frozen") is not True
        ):
            raise ValueError("El checkpoint inicial debe ser de una cabeza ResNet18 congelada con los mismos datos y preprocesamiento.")
        model.load_state_dict(initial["state_dict"])
        initialization = {"sha256": hashlib.sha256(content).hexdigest(), "epoch": initial["epoch"]}
        del initial, content
    model.to(device.device)
    initial_weight = model.fc_out.weight.detach().clone()
    initial_bias = model.fc_out.bias.detach().clone()
    validation_loss = nn.BCEWithLogitsLoss()
    train_loss_fn = validation_loss
    train_loss_name = "BCE"
    if config.focal_loss:
        train_loss_fn = partial(sigmoid_focal_loss, alpha=-1, gamma=2, reduction="mean")
        train_loss_name = "focal (γ=2)"
    elif config.positive_weight != 1.0:
        train_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.positive_weight, device=device.device),
        )
        train_loss_name = "BCE ponderada"
    if config.label_smoothing:
        criterion = train_loss_fn
        smoothing = config.label_smoothing

        def train_loss_fn(logits, labels):
            return criterion(logits, labels * (1 - smoothing) + 0.5 * smoothing)

        train_loss_name += " con suavizado"
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_groups = trainable
    if config.backbone_learning_rate:
        head = list(model.fc_out.parameters())
        head_ids = {id(parameter) for parameter in head}
        parameter_groups = [
            {"params": head, "lr": config.learning_rate},
            {"params": [parameter for parameter in trainable if id(parameter) not in head_ids],
             "lr": config.backbone_learning_rate},
        ]
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    optimizer_options = {"momentum": config.momentum} if config.optimizer == "sgd" else {}
    optimizer = optimizers[config.optimizer](
        parameter_groups, lr=config.learning_rate, weight_decay=config.weight_decay, **optimizer_options,
    )
    scheduler = None
    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.progressive_head_epochs:
        model.configure_tuning(enabled=False)
    scaler = device.make_grad_scaler()
    stopping = EarlyStopping(config.patience)
    history: list[dict] = []
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "architecture": {"name": config.architecture, "image_size": config.image_size, "dropout": config.dropout},
        "config": asdict(config), "optimizer": type(optimizer).__name__, "seed": config.seed,
        "effective_batch_size": config.batch_size * config.accumulation_steps,
        "normalization": asdict(loaders.normalization), "manifest_sha256": digest,
        "classes": CLASSES, "torch_version": str(torch.__version__),
        "preprocessing": {
            "color_mode": "RGB", "interpolation": "bilinear", "pixel_range": [0, 1],
            "geometry": "letterbox" if transfer_learning else "resize_square",
            "normalization_source": "ImageNet" if transfer_learning else "PAPILA train",
        },
        "device": str(device.device), "amp_fp16": device.amp_enabled and device.amp_dtype == torch.float16,
        "precision": str(device.amp_dtype).removeprefix("torch.") if device.amp_enabled else "float32",
        "pin_memory": device.pin_memory,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if transfer_learning:
        weights = BACKBONES[config.architecture][1]
        metadata["pretrained_weights"] = {
            "enum": str(weights), "url": weights.url, "sha256": WEIGHT_DIGESTS[config.architecture],
            "backbone_frozen": not config.tuned_stages,
        }
        if config.tuned_stages:
            if config.architecture == EFFICIENTNET_B0:
                block = f"features[-{config.tuned_stages + 1}:]"
            else:
                block = "layer4" if config.tuned_stages == 1 else "layer3+layer4"
            metadata["pretrained_weights"].update(
                trainable_block=block, batch_norm_frozen=config.freeze_batch_norm,
            )
        metadata["preprocessing"].update(padding_fill=0, centering=[0.5, 0.5])
    if initialization is not None:
        metadata["initial_checkpoint"] = initialization
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "best.pt"
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            if config.progressive_head_epochs and epoch == config.progressive_head_epochs + 1:
                model.configure_tuning(enabled=True)
                stopping.stale_epochs = 0
            try:
                train_loss, skipped = _train_epoch(
                    model, loaders.train, device, train_loss_fn, optimizer, scaler, config.accumulation_steps,
                )
                validation = _validate(model, loaders.val, device, validation_loss, config.threshold)
            except FloatingPointError as error:
                raise FloatingPointError(f"Época {epoch}: {error}") from error
            row = {
                "epoch": epoch, "train_loss": train_loss, **validation, "amp_skipped_steps": skipped,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "head_only_warmup": epoch <= config.progressive_head_epochs,
                "head_lr": optimizer.param_groups[0]["lr"], "last_group_lr": optimizer.param_groups[-1]["lr"],
            }
            if not history:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
            improved, should_stop = stopping.step(validation["val_loss"])
            if config.scheduler == "plateau":
                scheduler.step(validation["val_loss"])
            elif scheduler is not None:
                scheduler.step()
            if improved:
                _save_checkpoint(checkpoint_path, {
                    **metadata, "epoch": epoch, "best_val_loss": stopping.best_loss, "validation": validation,
                    "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                })
            if should_stop and epoch > config.progressive_head_epochs:
                break

    selected = read_checkpoint(checkpoint_path)
    model.load_state_dict(selected["state_dict"])
    if torch.equal(model.fc_out.weight, initial_weight) and torch.equal(model.fc_out.bias, initial_bias):
        raise FloatingPointError("Los pesos de salida restaurados no cambiaron respecto a la inicialización.")
    _plot_loss(history, selected["epoch"], run_dir / "loss.png", config.architecture, train_loss_name)
    _plot_validation(history, selected["epoch"], run_dir / "validation.png", config.threshold)
    summary = {
        **metadata, "checkpoint": str(checkpoint_path), "epochs_run": len(history),
        "best_epoch": selected["epoch"], "best_val_loss": selected["best_val_loss"],
        "validation": selected["validation"], "early_stopped": should_stop,
        "train_images": len(loaders.train.dataset), "val_images": len(loaders.val.dataset),
        "amp_skipped_steps": sum(row["amp_skipped_steps"] for row in history), "updated_head": True,
        "gpu": torch.cuda.get_device_name(device.device) if use_cuda else None,
        "cuda_version": torch.version.cuda if use_cuda else None,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device.device) / 2**20 if use_cuda else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device.device) / 2**20 if use_cuda else None,
        "seconds": time.perf_counter() - started,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    for name, default in asdict(defaults).items():
        options = {"action": argparse.BooleanOptionalAction} if isinstance(default, bool) else {"type": type(default)}
        if name == "architecture":
            options["choices"] = (ARCHITECTURE, *BACKBONES)
        parser.add_argument(
            f"--{name.replace('_', '-')}", default=None if name in {"epochs", "image_size"} else default, **options,
        )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    try:
        transfer_learning = args.architecture in BACKBONES
        if args.epochs is None:
            args.epochs = 30 if transfer_learning else defaults.epochs
        if args.image_size is None:
            args.image_size = 224 if transfer_learning else defaults.image_size
        run_dir = args.run_dir or Path("models/runs") / (f"papila-{args.architecture}" if transfer_learning else "papila-reference")
        config = TrainConfig(**{name: getattr(args, name) for name in asdict(defaults)})
        # Set the cuBLAS configuration before selecting/initializing CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        device = select_device(prefer_cuda=not args.cpu, enable_amp=not args.no_amp,
                               amp_dtype=dtype, pin_memory=args.pin_memory)
        summary = run_training(args.dataset_dir, args.manifest, run_dir, config, device,
                               init_checkpoint=args.init_checkpoint)
    except torch.OutOfMemoryError as error:
        parser.exit(1, f"Memoria CUDA insuficiente. Reduce --batch-size y usa otra --run-dir. {error}\n")
    except (FloatingPointError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Entrenamiento interrumpido: {error}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
