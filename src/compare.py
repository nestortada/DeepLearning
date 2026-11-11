"""Compare the CNN and frozen ResNet18 on the same PAPILA validation images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from .data import PapilaDataset, _read_manifest
from .device import select_device
from .model import ARCHITECTURE, load_checkpoint
from .predict import checkpoint_transform
from .pretrained import RESNET18
from .train import _seed_run, classification_metrics, validation_metrics, validation_outputs


THRESHOLDS = tuple(step / 20 for step in range(21))
MIN_SPECIFICITY = 0.80


def analyze_thresholds(labels: torch.Tensor, scores: torch.Tensor) -> list[dict]:
    """Compute the fixed 0.00–1.00 grid on validation scores."""
    return [
        {"threshold": threshold, **classification_metrics(labels, scores >= threshold)}
        for threshold in THRESHOLDS
    ]


def select_threshold(rows: list[dict]) -> dict | None:
    """Maximize sensitivity above the floor, then specificity and threshold."""
    return max(
        (row for row in rows if row["specificity"] >= MIN_SPECIFICITY),
        key=lambda row: (row["recall"], row["specificity"], row["threshold"]),
        default=None,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(
    output: Path, labels: torch.Tensor, scores: dict[str, torch.Tensor],
    areas: dict[str, tuple[float, float]], *, split_label: str,
) -> None:
    figure = Figure(figsize=(10, 4.6), layout="constrained")
    FigureCanvasAgg(figure)
    roc, pr = figure.subplots(1, 2)
    for name, logits in scores.items():
        roc_auc, average_precision = areas[name]
        false_positive_rate, sensitivity, _ = roc_curve(labels, logits)
        precision, recall, _ = precision_recall_curve(labels, logits)
        roc.plot(false_positive_rate, sensitivity, label=f"{name} · AUC {roc_auc:.3f}")
        pr.step(recall, precision, where="post", label=f"{name} · AP {average_precision:.3f}")
    roc.plot([0, 1], [0, 1], "--", color="gray")
    pr.axhline(labels.float().mean().item(), linestyle="--", color="gray", label="Fracción positiva")
    roc.set(title=f"ROC · {split_label}", xlabel="1 − especificidad", ylabel="Sensibilidad")
    pr.set(title=f"Precisión–recall · {split_label}", xlabel="Sensibilidad", ylabel="Precisión")
    for axis in (roc, pr):
        axis.set(xlim=(0, 1.02), ylim=(0, 1.02))
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.savefig(output, dpi=150)


def compare_validation(
    dataset_dir: Path, manifest: Path, cnn_run: Path, resnet_run: Path, output: Path, *, cpu: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError(f"La carpeta de comparación ya existe: {output}")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    records = _read_manifest(manifest)["val"]
    labels = torch.tensor([record.label for record in records], dtype=torch.float32)
    if labels.unique().numel() != 2:
        raise ValueError("Se requieren ambas clases de validación para las curvas y el análisis de umbral.")
    _seed_run(42)
    device = select_device(prefer_cuda=not cpu, enable_amp=False)
    models, logits_by_model = {}, {}
    thresholds = []
    for name, architecture, run in (("CNN", ARCHITECTURE, cnn_run), ("ResNet18", RESNET18, resnet_run)):
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        checkpoint_path = run / "best.pt"
        model, checkpoint = load_checkpoint(checkpoint_path, device.device)
        if checkpoint["architecture"]["name"] != architecture or summary["architecture"]["name"] != architecture:
            raise ValueError(f"Arquitectura inesperada en {run}")
        if checkpoint["manifest_sha256"] != digest or summary["manifest_sha256"] != digest:
            raise ValueError(f"El manifiesto de {name} difiere del manifiesto de comparación.")
        if checkpoint["epoch"] != summary["best_epoch"]:
            raise ValueError(f"El resumen y el checkpoint de {name} registran épocas distintas.")
        dataset = PapilaDataset(dataset_dir, records, checkpoint_transform(checkpoint))
        loader = DataLoader(dataset, batch_size=checkpoint["config"]["batch_size"], pin_memory=device.pin_memory)
        loss, observed_labels, logits = validation_outputs(model, loader, device, torch.nn.BCEWithLogitsLoss())
        if not torch.equal(observed_labels, labels):
            raise ValueError(f"El orden de etiquetas de {name} difiere del manifiesto.")
        logits_by_model[name] = logits
        rows = analyze_thresholds(labels, logits.sigmoid())
        thresholds.extend({"model": name, **row} for row in rows)
        models[name] = {
            "checkpoint": str(checkpoint_path), "selected_epoch": checkpoint["epoch"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "validation_at_0_5": {"val_loss": loss, **validation_metrics(labels, logits, 0.5)},
            "exploratory_threshold": select_threshold(rows),
            "trainable_parameters": summary.get(
                "trainable_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad),
            ),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "training": {field: summary[field] for field in (
                "epochs_run", "seconds", "gpu", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib",
            )},
        }
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
        raise ValueError("El manifiesto cambió durante la comparación.")

    output.mkdir(parents=True)
    scores = {name: logits.sigmoid().tolist() for name, logits in logits_by_model.items()}
    _write_csv(output / "predictions.csv", [
        {"image_id": record.image_id, "patient_id": record.patient_id, "label": record.label,
         "cnn_score": scores["CNN"][i], "resnet18_score": scores["ResNet18"][i]}
        for i, record in enumerate(records)
    ])
    _write_csv(output / "thresholds.csv", thresholds)
    areas = {
        name: (result["validation_at_0_5"]["val_roc_auc"], result["validation_at_0_5"]["val_average_precision"])
        for name, result in models.items()
    }
    plot_curves(output / "curves.png", labels, logits_by_model, areas, split_label="validación")
    report = {
        "split": "val", "manifest_sha256": digest, "images": len(records),
        "patients": len({record.patient_id for record in records}), "positives": int(labels.sum().item()),
        "comparison_threshold": 0.5,
        "exploratory_selection": {"grid": THRESHOLDS, "minimum_specificity": MIN_SPECIFICITY,
                                  "order": ["recall", "specificity", "threshold"]},
        "models": models,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cnn-run", type=Path, default=Path("models/runs/papila-reference"))
    parser.add_argument("--resnet-run", type=Path, default=Path("models/runs/papila-resnet18"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    try:
        report = compare_validation(args.dataset_dir, args.manifest, args.cnn_run, args.resnet_run,
                                    args.output or args.resnet_run / "comparison", cpu=args.cpu)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Comparación interrumpida: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
