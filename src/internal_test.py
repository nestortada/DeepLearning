"""Evaluate two frozen PAPILA checkpoints on the reserved test split once."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .compare import _write_csv, plot_curves
from .data import ImageRecord, PapilaDataset, _read_manifest
from .device import DeviceConfig, select_device
from .model import CLASSES, load_checkpoint
from .predict import checkpoint_transform
from .train import _seed_run


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "papila-internal-test.json"
OUTPUT = ROOT / "models" / "runs" / "papila-internal-test"
SCOPE = "Evaluación interna exploratoria de PAPILA con consulta histórica del conjunto de prueba."
METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "f2", "roc_auc", "average_precision", "fp", "fn")
DEPENDENCIES = ("torch", "torchvision", "numpy", "scipy", "Pillow", "scikit-learn", "matplotlib")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cohort_identity(records: list[ImageRecord], dataset_dir: Path) -> dict:
    """Fingerprint sorted image identities and JPEG bytes without decoding pixels."""
    patients = sorted({record.patient_id for record in records})
    inventory = [
        [record.image_id, record.patient_id, record.eye, record.label,
         record.relative_path.as_posix(), sha256_file(dataset_dir / record.relative_path)]
        for record in sorted(records, key=lambda row: row.image_id)
    ]
    positives = sum(record.label for record in records)
    return {
        "images": len(records), "patients": len(patients), "positives": positives,
        "negatives": len(records) - positives,
        "patient_ids_sha256": hashlib.sha256(json.dumps(patients, separators=(",", ":")).encode()).hexdigest(),
        "image_inventory_sha256": hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest(),
    }


def checkpoint_metadata(checkpoint: dict) -> dict:
    return {key: checkpoint.get(key) for key in (
        "format_version", "architecture", "epoch", "classes", "preprocessing", "normalization", "pretrained_weights",
    )}


def dependency_versions() -> dict:
    return {"python": platform.python_version(), **{name: version(name) for name in DEPENDENCIES}}


def _prepare(config: dict, dataset_dir: Path, manifest: Path, *, preflight: bool):
    if dependency_versions() != config["dependencies"]:
        raise ValueError("Las versiones de dependencias difieren de la configuración congelada.")
    digest = config["dataset"]["manifest_sha256"]
    if sha256_file(manifest) != digest:
        raise ValueError("SHA-256 del manifiesto distinto del congelado.")
    records = _read_manifest(manifest)
    for split in records if preflight else ("test",):
        if cohort_identity(records[split], dataset_dir) != config["dataset"]["splits"][split]:
            raise ValueError(f"Identidad de imágenes o pacientes distinta para {split}.")
    if sha256_file(manifest) != digest:
        raise ValueError("El manifiesto cambió durante la verificación.")

    _seed_run(config["execution"]["seed"])
    device = select_device(enable_amp=False)
    expected = config["execution"]
    if (str(device.device) != expected["device"] or torch.version.cuda != expected["cuda_version"]
            or torch.cuda.get_device_name(device.device) != expected["gpu"]):
        raise ValueError("El dispositivo o la versión CUDA difieren de la configuración congelada.")

    models = {}
    for name, spec in config["models"].items():
        path = ROOT / spec["checkpoint"]
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"SHA-256 del checkpoint de {name} distinto del congelado.")
        model, checkpoint = load_checkpoint(path, device.device)
        metadata = json.loads(json.dumps(checkpoint_metadata(checkpoint)))
        if metadata != spec["metadata"] or checkpoint["manifest_sha256"] != digest:
            raise ValueError(f"Arquitectura, preprocesamiento o procedencia incorrectos para {name}.")
        models[name] = (model.requires_grad_(False), checkpoint_transform(checkpoint))
    return records["test"], device, models


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def _forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if any(module.training for module in model.modules()):
        raise ValueError("Todos los módulos deben estar en modo evaluación.")
    if torch.is_grad_enabled() or any(parameter.requires_grad or parameter.grad is not None
                                     for parameter in model.parameters()):
        raise ValueError("Los gradientes deben estar deshabilitados antes de evaluar.")
    logits = model(images)
    if logits.shape != (images.shape[0],) or logits.dtype != torch.float32 or not logits.isfinite().all().item():
        raise ValueError("Se esperaba un logit FP32 finito por imagen.")
    return logits.cpu()


def _score(model: nn.Module, loader: DataLoader, device: DeviceConfig) -> tuple[np.ndarray, np.ndarray]:
    before = _state_hash(model)
    logits, labels = [], []
    for images, targets in loader:
        images = images.to(device.device, non_blocking=device.pin_memory)
        logits.append(_forward(model, images))
        labels.append(targets)
    if _state_hash(model) != before:
        raise ValueError("Los parámetros o buffers cambiaron durante la evaluación.")
    return torch.cat(labels).numpy().astype(int), torch.cat(logits).numpy()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def binary_metrics(labels: np.ndarray, logits: np.ndarray, predicted: np.ndarray) -> dict:
    tn, fp, fn, tp = map(int, np.bincount(2 * labels + predicted, minlength=4))
    positives, negatives = tp + fn, tn + fp
    return {
        "accuracy": (tp + tn) / len(labels), "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, positives), "specificity": _ratio(tn, negatives),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn) if positives else None,
        "f2": _ratio(5 * tp, 5 * tp + fp + 4 * fn) if positives else None,
        "roc_auc": float(roc_auc_score(labels, logits)) if positives and negatives else None,
        "average_precision": float(average_precision_score(labels, logits)) if positives and negatives else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _difference(a: int | float | None, b: int | float | None) -> int | float | None:
    return a - b if a is not None and b is not None else None


def bootstrap_intervals(
    patient_ids: list[str], labels: np.ndarray, logits: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray], estimates: dict, *, replicates: int, seed: int, confidence: float,
    pair: tuple[str, str] = ("ResNet18", "CNN"),
) -> dict:
    groups = defaultdict(list)
    for index, patient in enumerate(patient_ids):
        groups[patient].append(index)
    clusters = [groups[patient] for patient in sorted(groups)]
    rng = np.random.default_rng(seed)
    candidate, reference = pair
    difference = f"{candidate}_minus_{reference}"
    samples = {name: {metric: [] for metric in METRICS} for name in (*logits, difference)}
    for draw in rng.integers(len(clusters), size=(replicates, len(clusters))):
        indices = np.concatenate([clusters[index] for index in draw])
        values = {name: binary_metrics(labels[indices], score[indices], predictions[name][indices])
                  for name, score in logits.items()}
        values[difference] = {metric: _difference(values[candidate][metric], values[reference][metric])
                              for metric in METRICS}
        for name, metrics in values.items():
            for metric in METRICS:
                if metrics[metric] is not None:
                    samples[name][metric].append(metrics[metric])
    tail = (1 - confidence) / 2
    intervals = {}
    for name, metrics in samples.items():
        intervals[name] = {}
        for metric, valid in metrics.items():
            lower, upper = map(float, np.quantile(valid, (tail, 1 - tail))) if valid else (None, None)
            intervals[name][metric] = {
                "estimate": estimates[name][metric], "lower": lower, "upper": upper,
                "valid_resamples": len(valid), "invalid_resamples": replicates - len(valid),
            }
    return intervals


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _plot_confusion(path: Path, matrices: dict) -> None:
    figure = Figure(figsize=(8, 3.9), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (name, matrix) in zip(figure.subplots(1, 2), matrices.items(), strict=True):
        ConfusionMatrixDisplay(np.asarray(matrix), display_labels=CLASSES).plot(ax=axis, colorbar=False)
        axis.set(title=name, xlabel="Predicción", ylabel="Etiqueta real")
    figure.savefig(path, dpi=150)


def run_preflight(config: dict, dataset_dir: Path, manifest: Path) -> dict:
    _, device, models = _prepare(config, dataset_dir, manifest, preflight=True)
    for name, (model, _) in models.items():
        size = config["models"][name]["metadata"]["architecture"]["image_size"]
        before = _state_hash(model)
        _forward(model, torch.zeros((1, 3, size, size), device=device.device))
        if _state_hash(model) != before:
            raise ValueError(f"El modelo {name} cambió en la prueba sintética.")
    return {"status": "preflight_ok", "test_images_decoded": 0, "device": str(device.device),
            "splits": config["dataset"]["splits"], "config_sha256": sha256_file(CONFIG_PATH)}


def run_test(config: dict, dataset_dir: Path, manifest: Path) -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"La evaluación ya tiene una carpeta reservada: {OUTPUT}")
    records, device, models = _prepare(config, dataset_dir, manifest, preflight=False)
    reproducibility = {
        "status": "started", "scope": SCOPE, "started_utc": datetime.now(UTC).isoformat(),
        "config_sha256": sha256_file(CONFIG_PATH), "dependencies": dependency_versions(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(device.device),
        "cudnn_version": torch.backends.cudnn.version(), "execution": config["execution"],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "frozen_config.json").write_bytes(CONFIG_PATH.read_bytes())
    _json(OUTPUT / "reproducibility.json", reproducibility)

    labels = np.asarray([record.label for record in records], dtype=int)
    logits = {}
    for name, (model, transform) in models.items():
        loader = DataLoader(PapilaDataset(dataset_dir, records, transform),
                            batch_size=config["execution"]["batch_size"], pin_memory=device.pin_memory)
        observed, logits[name] = _score(model, loader, device)
        if not np.array_equal(labels, observed):
            raise ValueError(f"Las etiquetas de {name} no coinciden con el orden congelado.")
    scores = {name: torch.from_numpy(values).sigmoid().numpy() for name, values in logits.items()}
    thresholds = {name: spec["threshold"] for name, spec in config["models"].items()}
    predictions = {name: values >= thresholds[name] for name, values in scores.items()}
    images, patients = [], defaultdict(list)
    for index, record in enumerate(records):
        row = {"image_id": record.image_id, "eye": record.eye, "label": record.label}
        for name in logits:
            row.update({f"{name}_logit": float(logits[name][index]), f"{name}_score": float(scores[name][index]),
                        f"{name}_prediction": int(predictions[name][index])})
        images.append({"patient_id": record.patient_id, **row})
        patients[record.patient_id].append(row)
    _write_csv(OUTPUT / "predictions_images.csv", images)
    _json(OUTPUT / "predictions_patients.json", {"patients": dict(patients)})

    test = {name: binary_metrics(labels, score, predictions[name]) for name, score in logits.items()}
    validation = {name: spec["validation_metrics"] for name, spec in config["models"].items()}
    paired = {metric: _difference(test["ResNet18"][metric], test["CNN"][metric]) for metric in METRICS}
    intervals = bootstrap_intervals([record.patient_id for record in records], labels, logits, predictions,
                                    {**test, "ResNet18_minus_CNN": paired}, **config["bootstrap"])
    matrices = {name: [[m["tn"], m["fp"]], [m["fn"], m["tp"]]] for name, m in test.items()}
    report = {
        "scope": SCOPE, "test_cohort": config["dataset"]["splits"]["test"], "thresholds": thresholds,
        "test_metrics": test, "validation_metrics_at_same_threshold": validation,
        "test_minus_validation": {name: {metric: _difference(test[name][metric], validation[name][metric])
                                         for metric in METRICS} for name in test},
        "confusion_matrices_order_sano_glaucoma": matrices,
        "resnet18_minus_cnn_test": paired, "bootstrap": config["bootstrap"], "intervals": intervals,
    }
    _json(OUTPUT / "metrics.json", report)
    _write_csv(OUTPUT / "metrics.csv", [
        {"split": split, "model": name, "threshold": thresholds[name], **{key: result[key] for key in METRICS},
         **{key: result[key] for key in ("tn", "tp")}}
        for split, values in (("validation", validation), ("test", test)) for name, result in values.items()
    ])
    _write_csv(OUTPUT / "confidence_intervals.csv", [
        {"model": name, "metric": metric, **interval} for name, items in intervals.items()
        for metric, interval in items.items()
    ])
    _plot_confusion(OUTPUT / "confusion_matrices.png", matrices)
    plot_curves(OUTPUT / "roc_pr.png", torch.from_numpy(labels),
                {name: torch.from_numpy(values) for name, values in logits.items()},
                {name: (m["roc_auc"], m["average_precision"]) for name, m in test.items()}, split_label="test interno")
    _json(OUTPUT / "reproducibility.json", {**reproducibility, "status": "complete",
          "completed_utc": datetime.now(UTC).isoformat(), "model_state_unchanged": True})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Verifica identidad y prueba tensores sintéticos.")
    mode.add_argument("--confirm-internal-test", action="store_true", help="Confirma la única evaluación de test.")
    args = parser.parse_args()
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        operation = run_preflight if args.preflight else run_test
        report = operation(config, args.dataset_dir, args.manifest)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Evaluación interrumpida: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
