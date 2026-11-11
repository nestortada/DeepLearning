"""Evaluate PAPILA research runs on validation and freeze the final test comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import time
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import transforms

from src.compare import _write_csv, plot_curves
from src.data import PapilaDataset, _read_manifest
from src.device import select_device
from src.internal_test import (
    CONFIG_PATH, METRICS, _difference, _json, _score, bootstrap_intervals, cohort_identity,
    dependency_versions, sha256_file,
)
from src.model import CLASSES, load_checkpoint
from src.predict import checkpoint_transform
from src.train import classification_metrics, validation_metrics

from .train_research import phase_runs


PRODUCTION_CHECKPOINT = Path("backend/predict/best.pt")
WEIGHTED_CHECKPOINT = Path("models/runs/papila-loss-20261109/bce-ponderada/best.pt")
WEIGHTED_SHA256 = "6a009c9d84a737768ccaf6f97b0914025da2adb46e95ff62d1d32fa5a3b04232"
FOCAL_CHECKPOINT = Path("models/runs/papila-loss-20261109/focal/best.pt")
ARCHIVED_WEIGHTS = Path("models/runs/papila-weights-20261109/weight-sweep.json")
AGGREGATE_METRICS = ("recall", "specificity", "balance", "balanced_accuracy", "f2", "roc_auc")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def balanced_metrics(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> dict:
    row = classification_metrics(labels, scores.double() >= threshold)
    recall, specificity = row["recall"], row["specificity"]
    return {"threshold": threshold, **row, "balance": min(recall, specificity),
            "balanced_accuracy": (recall + specificity) / 2}


def threshold_sweep(labels: torch.Tensor, scores: torch.Tensor, target: float) -> tuple[dict, list[dict]]:
    """Use midpoints between distinct validation scores for each attainable decision."""
    if labels.unique().tolist() != [0, 1] or not scores.isfinite().all():
        raise ValueError("La selección del umbral requiere ambas clases y puntajes finitos.")
    values = scores.double().unique(sorted=True).tolist()
    midpoints = [(left + right) / 2 for left, right in zip(values, values[1:])]
    # If a score saturates at 1, predicting all negatives is impossible under >=.
    rows = [balanced_metrics(labels, scores, threshold) for threshold in (0.0, *midpoints, 1.0)]
    best = max(rows, key=lambda row: (
        row["balance"] >= target, row["balance"], row["balanced_accuracy"], row["recall"], row["threshold"],
    ))
    return best, rows


def audit(dataset_dir: Path, manifest: Path, frozen: dict) -> dict:
    records = _read_manifest(manifest)
    if sha256_file(manifest) != frozen["manifest_sha256"]:
        raise ValueError("Manifiesto distinto del congelado.")
    splits = {split: cohort_identity(items, dataset_dir) for split, items in records.items()}
    if splits != frozen["splits"]:
        raise ValueError("El inventario de PAPILA difiere del congelado.")
    eye_pairs = {
        split: len(items) == 2 * len({item.patient_id for item in items})
        for split, items in records.items()
    }
    hashes = [sha256_file(dataset_dir / item.relative_path) for items in records.values() for item in items]
    duplicates = len(hashes) - len(set(hashes))
    if duplicates or not all(eye_pairs.values()):
        raise ValueError("Pares de ojos incompletos o JPEG duplicados.")
    return {"manifest_sha256": frozen["manifest_sha256"], "splits": splits,
            "two_eyes_per_patient": eye_pairs, "duplicate_jpegs": duplicates,
            "production_checkpoint_sha256": sha256_file(PRODUCTION_CHECKPOINT)}


def measure_latency(model: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> dict:
    sample = sample.unsqueeze(0).to(device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    durations = []
    with torch.inference_mode():
        for _ in range(8):
            model(sample)
        sync()
        for _ in range(30):
            start = time.perf_counter()
            model(sample)
            sync()
            durations.append((time.perf_counter() - start) * 1e3)
    return {"inference_batch1_ms_mean": statistics.mean(durations),
            "inference_batch1_ms_median": statistics.median(durations)}


def save_plots(output: Path, labels: torch.Tensor, logits: torch.Tensor, result: dict, split: str) -> None:
    plot_curves(output / "roc-pr.png", labels, {"Modelo": logits},
                {"Modelo": (result["roc_auc"], result["average_precision"])}, split_label=split)
    figure = Figure(figsize=(4, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
    axis = figure.subplots()
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=axis, colorbar=False)
    axis.set(title=f"{split} · umbral {result['threshold']:.4f}", xlabel="Predicción", ylabel="Etiqueta real")
    figure.savefig(output / "confusion.png", dpi=150)


def checkpoint_spec(path: Path, *, threshold: float | None = None, **info) -> dict:
    spec = {"checkpoint": str(path), "sha256": sha256_file(path), **info}
    if threshold is not None:
        spec["threshold"] = threshold
    return spec


def evaluate_model(
    dataset_dir: Path, records: list, manifest_digest: str, spec: dict, destination: Path,
    device, split: str, target: float, *, collect_latency: bool = True,
    cached_pixels: dict[int, torch.Tensor] | None = None,
) -> tuple[dict, torch.Tensor]:
    path = Path(spec["checkpoint"])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise ValueError(f"Checkpoint alterado: {path}")
    model, checkpoint = load_checkpoint(io.BytesIO(content), device.device)
    model.requires_grad_(False)
    if checkpoint["manifest_sha256"] != manifest_digest:
        raise ValueError(f"El checkpoint {path} se entrenó con otro manifiesto.")
    transform = checkpoint_transform(checkpoint)
    if cached_pixels is None:
        dataset = PapilaDataset(dataset_dir, records, transform)
    else:
        dataset = PapilaDataset(
            dataset_dir, records, transforms.Compose(transform.transforms[:2]),
            after_cache=transforms.Compose(transform.transforms[2:]), cached_pixels=cached_pixels,
        )
    loader = DataLoader(dataset, batch_size=16, pin_memory=device.pin_memory)
    observed, raw = _score(model, loader, device)
    labels, logits = torch.from_numpy(observed), torch.from_numpy(raw)
    if labels.tolist() != [item.label for item in records]:
        raise ValueError("Las etiquetas puntuadas difieren del manifiesto.")
    scores = logits.sigmoid()
    if split == "val" and "threshold" not in spec:
        operating, sweep = threshold_sweep(labels, scores, target)
    else:
        operating = balanced_metrics(labels, scores, spec["threshold"])
        sweep = []
    threshold = operating["threshold"]
    curves = validation_metrics(labels, logits, threshold)
    result = {
        **spec, **operating, "architecture": checkpoint["architecture"]["name"], "epoch": checkpoint["epoch"],
        "roc_auc": curves["val_roc_auc"], "average_precision": curves["val_average_precision"],
        "pr_auc": curves["val_pr_auc"], "target_met": operating["balance"] >= target,
        "eval_bce": torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float()).item(),
    }
    if collect_latency:
        result.update(measure_latency(model, dataset[0][0], device.device))
    destination.mkdir()
    _write_csv(destination / "predictions.csv", [
        {"image_id": item.image_id, "patient_id": item.patient_id, "label": item.label, "logit": logits[i].item(),
         "score": scores[i].item(), "prediction": int(scores[i].item() >= threshold)}
        for i, item in enumerate(records)
    ])
    if sweep:
        sweep.append({"comparison": "fixed_0.40", **balanced_metrics(labels, scores, 0.40)})
        _write_csv(destination / "thresholds.csv", [
            {"comparison": row.get("comparison", "midpoint"), **row, "roc_auc": result["roc_auc"]}
            for row in sweep
        ])
    save_plots(destination, labels, logits, result, split)
    _json(destination / "metrics.json", result)
    return result, logits


def aggregate_variants(rows: list[dict]) -> list[dict]:
    aggregated = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        group = [row for row in rows if row["variant"] == variant]
        entry = {"variant": variant, "architecture": group[0]["architecture"],
                 "seeds": len(group), "seeds_meeting_target": sum(row["target_met"] for row in group)}
        for metric in AGGREGATE_METRICS:
            values = [row[metric] for row in group]
            entry[f"mean_{metric}"] = statistics.mean(values)
            entry[f"sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else None
        aggregated.append(entry)
    return sorted(aggregated, key=lambda row: (
        row["seeds_meeting_target"], row["mean_balance"], row["mean_balanced_accuracy"],
        row["mean_recall"], row["mean_roc_auc"],
    ), reverse=True)


def training_record(run: Path, name: str, seed: int) -> dict:
    summary = read_json(run / "summary.json")
    gpu = read_json(run / "gpu-summary.json")
    with (run / "history.csv").open(encoding="utf-8", newline="") as stream:
        epochs = [float(row["epoch_seconds"]) for row in csv.DictReader(stream)]
    return checkpoint_spec(run / "best.pt", variant=name, seed=seed, **gpu, **{
        field: summary[field] for field in (
            "seconds", "epochs_run", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib",
            "trainable_parameters", "total_parameters", "precision", "pin_memory", "amp_skipped_steps",
        )
    }, epoch_seconds_mean=statistics.mean(epochs), epoch_seconds_first=epochs[0],
       epoch_seconds_warm_mean=statistics.mean(epochs[1:] or epochs))


def compare_phase(dataset_dir: Path, manifest: Path, campaign: Path, phase: str, device, frozen: dict) -> None:
    plan = read_json(campaign / "plan.json")
    target = plan["target"]
    val = _read_manifest(manifest)["val"]
    if cohort_identity(val, dataset_dir) != frozen["dataset"]["splits"]["val"]:
        raise ValueError("El inventario de validación difiere del congelado.")
    destination = campaign / f"{phase}-validation"
    if destination.exists():
        raise FileExistsError(f"Ya existe {destination}.")
    specs = {
        f"{name}-s{seed}": training_record(campaign / phase / f"{name}-s{seed}", name, seed)
        for name, _, seed, _ in phase_runs(plan, phase, campaign)
    }
    destination.mkdir()
    rows = []
    for key, spec in specs.items():
        result, _ = evaluate_model(dataset_dir, val, plan["manifest_sha256"], spec, destination / key,
                                   device, "val", target)
        rows.append(result)
    _write_csv(destination / "runs.csv", rows)
    if phase == "pilots":
        return
    if phase == "confirm":
        architectures = read_json(campaign / "architectures-validation" / "selection.json")
        screen = read_json(campaign / "screen-validation" / "selection.json")
        rows.extend(architectures["runs"])
        rows.extend(row for row in screen["runs"] if row["variant"] in screen["confirm_variants"])
    groups = aggregate_variants(rows)
    winner = groups[0]
    selection = {"phase": phase, "architecture": winner["architecture"], "winning_variant": winner["variant"],
                 "manifest_sha256": plan["manifest_sha256"], "target": target, "groups": groups, "runs": rows}
    if phase == "screen":
        selection["confirm_variants"] = [row["variant"] for row in groups[:plan["confirm_top"]]]
    _write_csv(destination / "variants.csv", groups)
    _json(destination / "selection.json", selection)


def verify_archived_weights(dataset_dir: Path, manifest: Path, campaign: Path, device, frozen: dict) -> None:
    archive = read_json(ARCHIVED_WEIGHTS)
    plan = read_json(campaign / "plan.json")
    digest = plan["manifest_sha256"]
    records = _read_manifest(manifest)["val"]
    if archive["manifest_sha256"] != digest or cohort_identity(records, dataset_dir) != frozen["dataset"]["splits"]["val"]:
        raise ValueError("La comparación de pesos archivada usa otra cohorte.")
    destination = campaign / "archive-validation"
    destination.mkdir()
    rows = []
    for weight in archive["weights"]:
        for member in weight["members"]:
            old_path = PureWindowsPath(member["checkpoint"]).parts
            checkpoint = Path(*old_path[old_path.index("models"):])
            spec = {"checkpoint": str(checkpoint), "sha256": member["sha256"],
                    "threshold": member["threshold"], "positive_weight": weight["positive_weight"],
                    "seed": member["seed"], "variant": member["name"]}
            result, _ = evaluate_model(dataset_dir, records, digest, spec, destination / member["name"],
                                       device, "val", plan["target"], collect_latency=False)
            for metric in ("recall", "specificity", "f2", "roc_auc"):
                if not np.isclose(result[metric], member[metric], atol=1e-8, rtol=0):
                    raise ValueError(f"{member['name']} no reproduce el valor archivado de {metric}.")
            rows.append(result)
    _write_csv(destination / "runs.csv", rows)
    _json(destination / "metrics.json", {
        "cohort": frozen["dataset"]["splits"]["val"], "archive_sha256": sha256_file(ARCHIVED_WEIGHTS),
        "groups": [
            {"positive_weight": weight["positive_weight"], "seeds": len(weight["members"]), **{
                metric: statistics.mean(row[metric] for row in rows if row["positive_weight"] == weight["positive_weight"])
                for metric in ("recall", "specificity", "f2", "roc_auc")
            }}
            for weight in archive["weights"]
        ],
    })


def lock_final(dataset_dir: Path, manifest: Path, campaign: Path, device, frozen: dict) -> None:
    confirmed = read_json(campaign / "confirm-validation" / "selection.json")
    target = confirmed["target"]
    digest = confirmed["manifest_sha256"]
    val = _read_manifest(manifest)["val"]
    if cohort_identity(val, dataset_dir) != frozen["dataset"]["splits"]["val"]:
        raise ValueError("El inventario de validación difiere del congelado.")
    variant = confirmed["winning_variant"]
    candidate = max((row for row in confirmed["runs"] if row["variant"] == variant), key=lambda row: (
        row["balance"], row["balanced_accuracy"], row["recall"], -row["eval_bce"], -row["seed"],
    ))
    baseline = next(row for row in confirmed["runs"] if row["variant"] == "resnet18" and row["seed"] == 42)
    # The legacy thresholds are fixed operational comparators. The new candidate uses only validation.
    specs = {
        "candidate": candidate, "resnet18-retrained": baseline,
        "weighted-archived": checkpoint_spec(WEIGHTED_CHECKPOINT, threshold=0.40),
        "demo": checkpoint_spec(PRODUCTION_CHECKPOINT, threshold=frozen["models"]["ResNet18"]["threshold"]),
    }
    if specs["demo"]["sha256"] != frozen["models"]["ResNet18"]["sha256"]:
        raise ValueError("Checkpoint publicado distinto del congelado.")
    if specs["weighted-archived"]["sha256"] != WEIGHTED_SHA256:
        raise ValueError("Checkpoint ponderado distinto del archivado.")
    # Keep only the fields required to reproduce the test evaluation.
    locked_specs = {
        key: {field: spec[field] for field in ("checkpoint", "sha256", "threshold")}
        for key, spec in specs.items()
    }
    destination = campaign / "locked-validation"
    destination.mkdir()
    for key, spec in locked_specs.items():
        evaluate_model(dataset_dir, val, digest, spec, destination / key, device, "val", target)
    if FOCAL_CHECKPOINT.is_file():
        evaluate_model(dataset_dir, val, digest, checkpoint_spec(FOCAL_CHECKPOINT),
                       destination / "focal-archived", device, "val", target)
    _json(campaign / "test-lock.json", {
        "manifest_sha256": digest, "target": target, "selection_sha256": sha256_file(
            campaign / "confirm-validation" / "selection.json"
        ), "candidate_variant": variant, "candidate_seed": candidate["seed"], "models": locked_specs,
        "validation": {key: read_json(destination / key / "metrics.json") for key in locked_specs},
        "environment": dependency_versions(),
    })


def run_final_test(dataset_dir: Path, manifest: Path, campaign: Path, device, frozen: dict) -> None:
    locked = read_json(campaign / "test-lock.json")
    digest = locked["manifest_sha256"]
    if locked["selection_sha256"] != sha256_file(campaign / "confirm-validation" / "selection.json"):
        raise ValueError("La selección de validación cambió después de fijar el umbral.")
    destination = campaign / "test"
    if destination.exists():
        raise FileExistsError(f"El test ya se reservó o evaluó en {destination}.")
    records = _read_manifest(manifest)["test"]
    cohort = cohort_identity(records, dataset_dir)
    if cohort != frozen["dataset"]["splits"]["test"]:
        raise ValueError("El inventario de test difiere del congelado.")
    for spec in locked["models"].values():
        if sha256_file(Path(spec["checkpoint"])) != spec["sha256"]:
            raise ValueError("Un checkpoint cambió después de fijar el umbral.")
    destination.mkdir()
    for key, spec in locked["models"].items():
        evaluate_model(dataset_dir, records, digest, spec, destination / key, device, "test", locked["target"])
    finalize_test(manifest, campaign, frozen)


def finalize_test(manifest: Path, campaign: Path, frozen: dict) -> None:
    """Summarize saved predictions without decoding or scoring test images again."""
    destination = campaign / "test"
    if (destination / "metrics.json").exists():
        raise FileExistsError(f"El informe de test ya existe en {destination}.")
    locked = read_json(campaign / "test-lock.json")
    if locked["manifest_sha256"] != sha256_file(manifest):
        raise ValueError("El manifiesto difiere del fijado para test.")
    if locked["selection_sha256"] != sha256_file(campaign / "confirm-validation" / "selection.json"):
        raise ValueError("La selección de validación cambió después de fijar el umbral.")
    records = _read_manifest(manifest)["test"]
    expected = [(item.image_id, item.patient_id, item.label) for item in records]
    rows, series, decisions = [], {}, {}
    for key, spec in locked["models"].items():
        result = read_json(destination / key / "metrics.json")
        if any(result[field] != value for field, value in spec.items()):
            raise ValueError(f"El resultado guardado difiere del bloqueo para {key}.")
        with (destination / key / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        if [(p["image_id"], p["patient_id"], int(p["label"])) for p in predictions] != expected:
            raise ValueError(f"Las predicciones guardadas difieren del manifiesto para {key}.")
        decisions[key] = np.array([int(p["prediction"]) for p in predictions])
        if not np.array_equal(decisions[key], [float(p["score"]) >= spec["threshold"] for p in predictions]):
            raise ValueError(f"Las predicciones guardadas difieren del umbral para {key}.")
        rows.append({"model": key, **result})
        series[key] = torch.tensor([float(p["logit"]) for p in predictions], dtype=torch.float32)
    labels = torch.tensor([item.label for item in records])
    areas = {row["model"]: (row["roc_auc"], row["average_precision"]) for row in rows}
    if not (destination / "roc-pr.png").exists():
        plot_curves(destination / "roc-pr.png", labels, series, areas, split_label="test interno")
    if not (destination / "metrics.csv").exists():
        _write_csv(destination / "metrics.csv", rows)
    observed = {row["model"]: row for row in rows}
    patient_ids = [item.patient_id for item in records]
    replicates = 2000
    confidence = 0.95
    comparisons = {}
    for reference in ("weighted-archived", "demo"):
        pair = ("candidate", reference)
        logits = {key: series[key].numpy() for key in pair}
        predictions = {key: decisions[key] for key in pair}
        estimates = {**observed, f"candidate_minus_{reference}": {
            metric: _difference(observed["candidate"][metric], observed[reference][metric]) for metric in METRICS
        }}
        comparisons[reference] = bootstrap_intervals(
            patient_ids, labels.numpy(), logits, predictions, estimates,
            replicates=replicates, seed=42, confidence=confidence, pair=pair,
        )
    _json(destination / "patient-bootstrap.json", {
        "replicates": replicates, "confidence": confidence, "seed": 42, "sampling_unit": "patient",
        "comparisons": comparisons,
    })
    _json(destination / "metrics.json", {
        "scope": "Test interno exploratorio consultado en campañas previas",
        "cohort": frozen["dataset"]["splits"]["test"],
        "lock_sha256": sha256_file(campaign / "test-lock.json"), "rows": rows,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(
        "audit", "architectures", "screen", "confirm", "pilots", "archive", "lock", "test", "finalize-test",
    ))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    frozen = read_json(CONFIG_PATH)
    if sha256_file(args.manifest) != frozen["dataset"]["manifest_sha256"]:
        raise ValueError("Manifiesto distinto del congelado.")
    if args.phase == "audit":
        output = args.campaign / "audit.json"
        if output.exists():
            raise FileExistsError(output)
        args.campaign.mkdir(parents=True, exist_ok=True)
        _json(output, audit(args.dataset_dir, args.manifest, frozen["dataset"]))
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    if args.phase == "finalize-test":
        finalize_test(args.manifest, args.campaign, frozen)
        return
    device = select_device(prefer_cuda=not args.cpu, enable_amp=False)
    if device.device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA no disponible. Usa --cpu para evaluar explícitamente en CPU.")
    if device.device.type == "cuda":
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    args_common = (args.dataset_dir, args.manifest, args.campaign)
    if args.phase == "lock":
        lock_final(*args_common, device, frozen)
    elif args.phase == "test":
        run_final_test(*args_common, device, frozen)
    elif args.phase == "archive":
        verify_archived_weights(*args_common, device, frozen)
    else:
        compare_phase(*args_common, args.phase, device, frozen)


if __name__ == "__main__":
    main()
