"""Summarize matched PAPILA seeds and score their prediction averages in validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from src.compare import _write_csv, analyze_thresholds, select_threshold
from src.internal_test import _json, binary_metrics


METRICS = ("recall", "specificity", "f2", "roc_auc")


def summarize(campaign: Path) -> dict:
    validation = json.loads((campaign / "validation.json").read_text(encoding="utf-8"))
    if validation["split"] != "val":
        raise ValueError("El resumen requiere predicciones de validación.")
    with (campaign / "val" / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = torch.tensor([int(row["label"]) for row in rows])

    groups = defaultdict(dict)
    for name, model in validation["models"].items():
        if name in ("CNN", "ResNet18"):
            continue
        config = model["training"]["config"]
        weight = float(config.get("positive_weight", 1.0))
        seed = config["seed"]
        if seed in groups[weight]:
            raise ValueError(f"Semilla repetida para el peso {weight}: {seed}")
        groups[weight][seed] = name

    seed_sets = {tuple(sorted(members)) for members in groups.values()}
    if len(groups) < 2 or len(seed_sets) != 1 or len(next(iter(seed_sets))) < 2:
        raise ValueError("Cada peso requiere el mismo conjunto de semillas, con al menos dos.")

    weights = []
    averages = {}
    for weight, group in sorted(groups.items()):
        names = [group[seed] for seed in sorted(group)]
        runs = [validation["models"][name] for name in names]
        stats = {}
        for metric in METRICS:
            values = [run[metric] for run in runs]
            stats[metric] = {"mean": mean(values), "min": min(values), "max": max(values)}

        member_scores = torch.tensor([[float(row[name]) for row in rows] for name in names])
        scores = member_scores.mean(dim=0)
        threshold = select_threshold(analyze_thresholds(labels, scores))["threshold"]
        ensemble = binary_metrics(labels.numpy(), scores.numpy(), (scores >= threshold).numpy())
        weights.append({
            "positive_weight": weight,
            "members": [
                {"name": name, "seed": run["training"]["config"]["seed"], "checkpoint": run["path"],
                 "sha256": run["sha256"], "threshold": run["threshold"],
                 **{metric: run[metric] for metric in METRICS}}
                for name, run in zip(names, runs, strict=True)
            ],
            "individual": stats,
            "ensemble": {"threshold": threshold, **ensemble},
        })
        averages[f"weight_{weight:.6g}"] = scores.tolist()

    report = {"split": "val", "manifest_sha256": validation["manifest_sha256"],
              "seeds": list(next(iter(seed_sets))), "weights": weights,
              "minimum_validation_specificity": validation["minimum_validation_specificity"]}
    _write_csv(campaign / "val" / "ensemble-predictions.csv", [
        {**{key: row[key] for key in ("patient_id", "image_id", "label")},
         **{name: scores[index] for name, scores in averages.items()}}
        for index, row in enumerate(rows)
    ])
    _json(campaign / "weight-sweep.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.campaign)
    for group in report["weights"]:
        print(f"Peso {group['positive_weight']:.4g} · "
              f"sensibilidad media {group['individual']['recall']['mean']:.3f} · "
              f"sensibilidad promedio de puntajes {group['ensemble']['recall']:.3f}")


if __name__ == "__main__":
    main()
