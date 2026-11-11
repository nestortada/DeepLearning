"""Compare PAPILA checkpoints and open the frozen test only for the selected candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.compare import MIN_SPECIFICITY, _write_csv, analyze_thresholds, plot_curves, select_threshold
from src.data import PapilaDataset, _read_manifest
from src.device import select_device
from src.internal_test import _json, _score, binary_metrics, cohort_identity, sha256_file
from src.model import CLASSES, load_checkpoint
from src.predict import checkpoint_transform
from src.train import _seed_run


FROZEN = Path(__file__).resolve().parents[1] / "configs" / "papila-internal-test.json"


def _run(run_dir: Path) -> dict:
    checkpoint = (run_dir / "best.pt").resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "training": summary}


def _latency(model: torch.nn.Module, image: torch.Tensor) -> float:
    """Measure a batch of one with the tensor already on the inference device."""
    with torch.inference_mode():
        for _ in range(5):
            model(image)
        torch.cuda.synchronize(image.device)
        started = time.perf_counter()
        for _ in range(30):
            model(image)
        torch.cuda.synchronize(image.device)
    return (time.perf_counter() - started) * 1000 / 30


def evaluate(dataset_dir: Path, manifest: Path, campaign: Path, runs: dict[str, Path], *, test: bool) -> dict:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    digest = sha256_file(manifest)
    split = "test" if test else "val"
    records = _read_manifest(manifest)[split]
    if digest != frozen["dataset"]["manifest_sha256"] or (
        cohort_identity(records, dataset_dir) != frozen["dataset"]["splits"][split]
    ):
        raise ValueError("La partición no coincide con el inventario congelado.")

    validation = campaign / "validation.json"
    destination = campaign / split
    if test:
        previous = json.loads(validation.read_text(encoding="utf-8"))
        selected = previous["selected"]
        if selected is None or previous["manifest_sha256"] != digest:
            raise ValueError("No hay candidato fijado en validación para este manifiesto.")
        models = {name: previous["models"][name] for name in ("CNN", "ResNet18", selected)}
    else:
        selected = None
        models = {name: _run(run) for name, run in runs.items()}
        for name in ("CNN", "ResNet18"):
            if models[name]["sha256"] != frozen["models"][name]["sha256"]:
                raise ValueError(f"La referencia {name} no coincide con su SHA-256 congelado.")

    _seed_run(frozen["execution"]["seed"])
    device = select_device(enable_amp=False)
    if device.device.type != "cuda":
        raise RuntimeError("Esta evaluación requiere CUDA.")
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    labels = torch.tensor([record.label for record in records])
    logits_by_model = {}
    destination.mkdir()
    metrics = {}
    thresholds = []
    for name, info in models.items():
        path = Path(info["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != info["sha256"]:
            raise ValueError(f"Cambió el checkpoint de {name}.")
        model, checkpoint = load_checkpoint(io.BytesIO(content), device.device)
        del content
        model.requires_grad_(False)
        training = info["training"]
        if (
            checkpoint["manifest_sha256"] != digest or training["manifest_sha256"] != digest
            or checkpoint["epoch"] != training["best_epoch"] or checkpoint["architecture"] != training["architecture"]
        ):
            raise ValueError(f"El checkpoint, manifiesto o resumen de {name} no coinciden.")
        data = PapilaDataset(dataset_dir, records, checkpoint_transform(checkpoint))
        loader = DataLoader(data, batch_size=16, pin_memory=device.pin_memory)
        observed, logits = _score(model, loader, device)
        if not torch.equal(torch.from_numpy(observed), labels):
            raise ValueError(f"El orden de etiquetas de {name} cambió.")
        scores = torch.from_numpy(logits).sigmoid()
        if test:
            threshold = info["threshold"] if name == selected else frozen["models"][name]["threshold"]
        else:
            rows = analyze_thresholds(labels, scores)
            thresholds.extend({"model": name, **row} for row in rows)
            threshold = select_threshold(rows)["threshold"]
        result = binary_metrics(labels.numpy(), logits, (scores >= threshold).numpy())
        image, _ = data[0]
        metrics[name] = {
            **info, "threshold": threshold, **result,
            "forward_ms_cuda": _latency(model, image.unsqueeze(0).to(device.device)),
        }
        if test and name != selected:
            validation_threshold = info["threshold"]
            metrics[name]["at_common_validation_rule"] = {
                "threshold": validation_threshold,
                **binary_metrics(labels.numpy(), logits, (scores >= validation_threshold).numpy()),
            }
        logits_by_model[name] = torch.from_numpy(logits)

    if not test:
        _write_csv(destination / "thresholds.csv", thresholds)
        baseline = metrics["ResNet18"]
        candidates = [
            name for name, result in metrics.items()
            if name not in frozen["models"] and result["recall"] > baseline["recall"]
            and result["f2"] > baseline["f2"]
        ]
        selected = max(candidates, key=lambda name: tuple(
            metrics[name][key] for key in ("recall", "f2", "specificity", "average_precision")
        ), default=None)
    report = {"split": split, "manifest_sha256": digest, "selected": selected,
              "minimum_validation_specificity": MIN_SPECIFICITY, "models": metrics}
    scores_by_model = {name: logits.sigmoid().tolist() for name, logits in logits_by_model.items()}
    _write_csv(destination / "predictions.csv", [
        {"patient_id": record.patient_id, "image_id": record.image_id, "label": record.label,
         **{name: scores[index] for name, scores in scores_by_model.items()}}
        for index, record in enumerate(records)
    ])
    plot_curves(destination / "roc-pr.png", labels, logits_by_model,
                {name: (m["roc_auc"], m["average_precision"]) for name, m in metrics.items()},
                split_label="test interno" if test else "validación")
    figure = Figure(figsize=(3 * len(metrics), 3.4), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (name, m) in zip(figure.subplots(1, len(metrics)), metrics.items(), strict=True):
        matrix = np.asarray([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(
            ax=axis, colorbar=False,
        )
        axis.set(title=name, xlabel="Predicción", ylabel="Etiqueta real")
    figure.savefig(destination / "confusion.png", dpi=150)
    _json(destination / "metrics.json", report)
    if not test:
        _json(validation, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "DIRECTORY"), default=[])
    parser.add_argument("--test", action="store_true", help="Evalúa únicamente el candidato fijado y las referencias.")
    args = parser.parse_args()
    runs = {name: Path(directory) for name, directory in args.run}
    if not args.test and (not {"CNN", "ResNet18"} <= runs.keys() or len(runs) < 3):
        parser.error("Indica las referencias CNN y ResNet18 y al menos un candidato mediante --run.")
    report = evaluate(args.dataset_dir, args.manifest, args.campaign, runs, test=args.test)
    print(f"Candidato fijado: {report['selected']}")


if __name__ == "__main__":
    main()
