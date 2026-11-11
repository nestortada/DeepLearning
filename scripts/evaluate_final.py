"""Lock the PAPILA finalist comparison and report patient-level uncertainty."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from src.compare import plot_curves
from src.data import _read_manifest
from src.device import select_device
from src.internal_test import CONFIG_PATH, _json, cohort_identity, sha256_file

from .evaluate_research import evaluate_model, read_json
from .train_final import ARCHIVE, PLAN, write_csv


BOOTSTRAP_METRICS = ("recall", "specificity", "precision", "f1", "f2", "roc_auc", "average_precision")


def selection_spec(row: dict) -> dict:
    return {field: row[field] for field in ("checkpoint", "sha256", "threshold")}


def lock(dataset_dir: Path, manifest: Path, campaign: Path, device, frozen: dict) -> dict:
    selection = campaign / "validation-selection.json"
    decision = read_json(selection)
    old = read_json(ARCHIVE / "test-lock.json")["models"]
    models = {"candidate": selection_spec(decision["winner"])}
    models.update({f"finalist-{index}": selection_spec(row)
                   for index, row in enumerate(decision["representatives"][1:], 2)})
    for arch in read_json(PLAN)["architectures"]:
        row = read_json(campaign / "architecture" / f"{arch}-last-s42" / "record.json")
        if row["status"] == "complete":
            models[f"{arch}-control"] = selection_spec(row)
    loss_cases = {
        "efficientnet-bce": ("loss", "efficientnet_b0-bce"),
        "efficientnet-weighted": ("architecture", "efficientnet_b0-head"),
        "efficientnet-focal": ("loss", "efficientnet_b0-focal"),
    }
    for key, (phase, recipe) in loss_cases.items():
        path = campaign / phase / f"{recipe}-s42" / "record.json"
        if path.is_file() and (row := read_json(path))["status"] == "complete":
            models[key] = selection_spec(row)
    models.update({"previous-candidate": old["candidate"], "weighted-archived": old["weighted-archived"],
                   "demo": old["demo"]})
    if models["demo"]["sha256"] != frozen["models"]["ResNet18"]["sha256"]:
        raise ValueError("La referencia publicada difiere de su hash congelado.")
    for spec in models.values():
        if sha256_file(Path(spec["checkpoint"])) != spec["sha256"]:
            raise ValueError(f"Checkpoint alterado: {spec['checkpoint']}")
    digest = sha256_file(manifest)
    if digest != decision["manifest_sha256"]:
        raise ValueError("Manifiesto distinto de la selección.")
    records = _read_manifest(manifest)["val"]
    if cohort_identity(records, dataset_dir) != frozen["dataset"]["splits"]["val"]:
        raise ValueError("Inventario de validación distinto.")
    report = {
        "selection_sha256": sha256_file(selection), "manifest_sha256": digest,
        "candidate_variant": decision["winner"]["recipe"], "candidate_seed": decision["winner"]["seed"],
        "models": models, "target": read_json(PLAN)["target"],
    }
    result_dir = campaign / "locked-validation"
    evaluate_split(dataset_dir, records, report, result_dir, device, "val")
    _json(campaign / "test-lock.json", report)
    return report


def evaluate_split(dataset_dir: Path, records: list, report: dict, output: Path, device, split: str) -> None:
    output.mkdir()
    shared_pixels = {}
    seen = {}
    for name, spec in report["models"].items():
        key = (spec["sha256"], spec["threshold"])
        path = output / name
        if key in seen:
            shutil.copytree(seen[key], path)
            metadata = read_json(path / "metrics.json")
            _json(path / "metrics.json", {**metadata, **spec})
        else:
            evaluate_model(dataset_dir, records, report["manifest_sha256"], spec, path,
                           device, split, report["target"], cached_pixels=shared_pixels)
            seen[key] = path


def weight_draws(patient_ids: list[str], replicates: int, seed: int) -> np.ndarray:
    ids, index = np.unique(patient_ids, return_inverse=True)
    groups = len(ids)
    rng = np.random.default_rng(seed)
    return rng.multinomial(groups, np.full(groups, 1 / groups), size=replicates)[:, index].astype(float)


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator != 0)


def bootstrap_values(labels: np.ndarray, logits: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict:
    positives = weights @ labels
    negatives = weights @ (1 - labels)
    tp = weights @ (labels * predicted)
    fp = weights @ ((1 - labels) * predicted)
    fn, tn = positives - tp, negatives - fp
    order = np.argsort(logits, kind="stable")
    sorted_logits = logits[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_logits) != 0) + 1]
    positive_bins = np.add.reduceat(weights[:, order] * labels[order], starts, axis=1)
    negative_bins = np.add.reduceat(weights[:, order] * (1 - labels[order]), starts, axis=1)
    pairs = np.sum(positive_bins * (np.cumsum(negative_bins, axis=1) - negative_bins / 2), axis=1)
    positive_desc = positive_bins[:, ::-1]
    tp_at_score = np.cumsum(positive_desc, axis=1)
    total_at_score = np.cumsum((positive_bins + negative_bins)[:, ::-1], axis=1)
    ap = np.nansum(positive_desc * divide(tp_at_score, total_at_score), axis=1)
    return {
        "recall": divide(tp, positives), "specificity": divide(tn, negatives),
        "precision": divide(tp, tp + fp), "f1": divide(2 * tp, 2 * tp + fn + fp),
        "f2": divide(5 * tp, 5 * tp + 4 * fn + fp), "roc_auc": divide(pairs, positives * negatives),
        "average_precision": divide(ap, positives),
    }


def interval(values: np.ndarray, estimate: float) -> dict:
    valid = values[np.isfinite(values)]
    lower, upper = np.quantile(valid, (0.025, 0.975)) if len(valid) else (None, None)
    return {"estimate": estimate, "lower": lower, "upper": upper, "valid_resamples": len(valid)}


def summarize_split(records: list, report: dict, folder: Path, split: str) -> dict:
    identities = [(item.image_id, item.patient_id, item.label) for item in records]
    labels = np.array([item.label for item in records])
    weights = weight_draws([item.patient_id for item in records], 4000, 42)
    rows, arrays, curves, areas = {}, {}, {}, {}
    for key, spec in report["models"].items():
        row = read_json(folder / key / "metrics.json")
        if selection_spec(row) != spec:
            raise ValueError(f"Métricas ajenas al bloqueo para {key}.")
        with (folder / key / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        if [(item["image_id"], item["patient_id"], int(item["label"])) for item in predictions] != identities:
            raise ValueError(f"Las predicciones de {key} difieren del manifiesto.")
        logits = np.array([float(item["logit"]) for item in predictions], dtype=np.float32)
        decisions = np.array([int(item["prediction"]) for item in predictions])
        if not np.array_equal(decisions, [float(item["score"]) >= spec["threshold"] for item in predictions]):
            raise ValueError(f"El umbral guardado difiere para {key}.")
        values = bootstrap_values(labels, logits, decisions, weights)
        arrays[key] = values
        rows[key] = {**row, "confidence_intervals": {field: interval(values[field], row[field])
                                                     for field in BOOTSTRAP_METRICS}}
        curves[key] = torch.from_numpy(logits)
        areas[key] = (row["roc_auc"], row["average_precision"])
    title = "validación" if split == "val" else "test interno exploratorio"
    for figure_name, names in (
        ("comparison", ("candidate", "finalist-2", "finalist-3", "previous-candidate", "resnet18-control", "demo")),
        ("loss", ("efficientnet-bce", "efficientnet-weighted", "efficientnet-focal")),
    ):
        displayed = [key for key in names if key in rows]
        if displayed:
            plot_curves(folder / f"{figure_name}-roc-pr.png", torch.from_numpy(labels),
                        {key: curves[key] for key in displayed}, {key: areas[key] for key in displayed},
                        split_label=title)

    def paired_intervals(first, second):
        return {
            field: interval(arrays[first][field] - arrays[second][field], rows[first][field] - rows[second][field])
            for field in BOOTSTRAP_METRICS
        }

    paired = {key: paired_intervals("candidate", key)
              for key in ("demo", "previous-candidate", "resnet18-control") if key in rows}
    loss_pair = ("efficientnet-bce", "efficientnet-weighted")
    return {"models": rows, "paired_candidate_minus_reference": paired,
            "paired_efficientnet_bce_minus_weighted": paired_intervals(*loss_pair) if all(key in rows for key in loss_pair) else None}


def verify_lock(manifest: Path, campaign: Path) -> dict:
    report = read_json(campaign / "test-lock.json")
    if sha256_file(manifest) != report["manifest_sha256"] or sha256_file(
        campaign / "validation-selection.json",
    ) != report["selection_sha256"]:
        raise ValueError("La selección o el manifiesto cambiaron tras el bloqueo.")
    return report


def finalize(manifest: Path, campaign: Path) -> None:
    report = verify_lock(manifest, campaign)
    cohorts = _read_manifest(manifest)
    output = campaign / "metrics.json"
    if output.exists():
        raise FileExistsError(output)
    sections = {
        split: summarize_split(cohorts[split], report, campaign / folder, split)
        for split, folder in (("val", "locked-validation"), ("test", "test"))
    }
    write_csv(campaign / "final_comparison.csv", [
        {"split": split, "model": key, **{field: value for field, value in metrics.items()
                                        if field != "confidence_intervals"}}
        for split, section in sections.items() for key, metrics in section["models"].items()
    ])
    _json(output, {
        "scope": "PAPILA interno exploratorio con consultas previas",
        "bootstrap": {"unit": "patient", "replicates": 4000, "seed": 42, "confidence": 0.95},
        "lock_sha256": sha256_file(campaign / "test-lock.json"), **sections,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("lock", "test", "finalize"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    if args.phase == "finalize":
        finalize(args.manifest, args.campaign)
        return
    frozen = read_json(CONFIG_PATH)
    device = select_device(enable_amp=False)
    if device.device.type != "cuda":
        raise RuntimeError("CUDA es necesario para esta evaluación.")
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    if args.phase == "lock":
        lock(args.dataset_dir, args.manifest, args.campaign, device, frozen)
    else:
        report = verify_lock(args.manifest, args.campaign)
        records = _read_manifest(args.manifest)["test"]
        if cohort_identity(records, args.dataset_dir) != frozen["dataset"]["splits"]["test"]:
            raise ValueError("Inventario de test distinto.")
        evaluate_split(args.dataset_dir, records, report, args.campaign / "test", device, "test")
        finalize(args.manifest, args.campaign)


if __name__ == "__main__":
    main()
