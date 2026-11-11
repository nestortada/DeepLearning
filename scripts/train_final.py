"""Execute the preregistered PAPILA screening, confirmation and hardware trials."""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import json
import os
import shutil
import statistics
import subprocess
import threading
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.data import _read_manifest
from src.device import select_device
from src.internal_test import CONFIG_PATH, _json, cohort_identity, dependency_versions, sha256_file
from src.train import TrainConfig, run_training

from .evaluate_research import checkpoint_spec, evaluate_model, read_json


PLAN = Path("configs/papila-final.json")
ARCHIVE = Path("models/runs/papila-research-20261110")
METRICS = ("recall", "specificity", "precision", "f1", "f2", "roc_auc", "average_precision",
           "balance", "balanced_accuracy")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_history(manifest: Path, frozen: dict) -> dict:
    """Match reported reference metrics to the saved predictions and exact artifacts."""
    reference = read_json(ARCHIVE / "test-lock.json")
    if sha256_file(manifest) != reference["manifest_sha256"]:
        raise ValueError("El manifiesto difiere del usado en la campaña anterior.")
    summary = read_json(ARCHIVE / "test" / "metrics.json")
    if summary["lock_sha256"] != sha256_file(ARCHIVE / "test-lock.json") or summary["cohort"] != frozen["dataset"]["splits"]["test"]:
        raise ValueError("El test archivado difiere de su bloqueo o inventario.")
    checks = {}
    for row in summary["rows"]:
        name = row["model"]
        spec = reference["models"][name]
        if sha256_file(Path(spec["checkpoint"])) != spec["sha256"] or row["threshold"] != spec["threshold"]:
            raise ValueError(f"Referencia archivada alterada: {name}")
        path = ARCHIVE / "test" / name / "predictions.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        labels = torch.tensor([int(item["label"]) for item in predictions])
        scores = torch.tensor([float(item["score"]) for item in predictions], dtype=torch.float64)
        from src.train import validation_metrics
        measured = validation_metrics(labels, torch.tensor([float(item["logit"]) for item in predictions]), spec["threshold"])
        from .evaluate_research import balanced_metrics
        operating = balanced_metrics(labels, scores, spec["threshold"])
        for field in ("tn", "fp", "fn", "tp", "recall", "specificity"):
            if operating[field] != row[field]:
                raise ValueError(f"Métrica archivada alterada: {name}/{field}")
        if abs(measured["val_roc_auc"] - row["roc_auc"]) > 1e-10:
            raise ValueError(f"ROC archivada alterada: {name}")
        checks[name] = {"checkpoint_sha256": spec["sha256"], "threshold": spec["threshold"],
                        **{key: row[key] for key in ("recall", "specificity", "roc_auc", "tn", "fp", "fn", "tp")}}
    pilot_paths = [ARCHIVE / "pilots" / f"{name}-s42" / "summary.json" for name in ("uncached", "fp32-cache")]
    hashes = [sha256_file(ARCHIVE / "pilots-validation" / f"{name}-s42" / "predictions.csv")
              for name in ("uncached", "fp32-cache")]
    if hashes[0] != hashes[1]:
        raise ValueError("Las predicciones de caché archivadas difieren.")
    prior_runs = sum(len(list((ARCHIVE / phase).glob("*/summary.json")))
                     for phase in ("architectures", "screen", "confirm", "pilots", "historical-reproduction"))
    prior_runs += int((ARCHIVE / "historical-reproduction" / "summary.json").is_file())
    return {"models": checks, "prior_completed_trainings": prior_runs,
            "cache_pilot_seconds": [read_json(path)["seconds"] for path in pilot_paths],
            "cache_pilot_predictions_sha256": hashes[0],
            "candidate_configuration": read_json(Path(reference["models"]["candidate"]["checkpoint"]).parent / "summary.json")["config"]}


def working_set_mib() -> float | None:
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("page_faults", ctypes.c_ulong),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                    ("quota_peak_paged", ctypes.c_size_t), ("quota_paged", ctypes.c_size_t),
                    ("quota_peak_nonpaged", ctypes.c_size_t), ("quota_nonpaged", ctypes.c_size_t),
                    ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]

    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(-1), ctypes.byref(counters), counters.size,
    )
    return counters.working_set / 2**20 if ok else None


@contextlib.contextmanager
def telemetry(output: Path):
    stop = threading.Event()
    memory = []

    def sample():
        while not stop.is_set():
            value = working_set_mib()
            if value is not None:
                memory.append(value)
            stop.wait(0.25)

    ram = threading.Thread(target=sample, daemon=True)
    ram.start()
    smi = shutil.which("nvidia-smi")
    process = None
    try:
        with (output / "gpu.csv").open("x", encoding="utf-8") as stream:
            if smi:
                process = subprocess.Popen([
                    smi, "--id=0", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits", "--loop-ms=500",
                ], stdout=stream, stderr=subprocess.DEVNULL)
            yield
    finally:
        stop.set()
        ram.join()
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        utilization, used = [], []
        with (output / "gpu.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) == 4 and row[1].strip().isdigit():
                    utilization.append(float(row[1]))
                if len(row) == 4 and row[2].strip().isdigit():
                    used.append(float(row[2]))
        _json(output / "resources.json", {
            "gpu_samples": len(utilization),
            "gpu_mean_percent": statistics.mean(utilization) if utilization else None,
            "gpu_peak_percent": max(utilization, default=None), "gpu_global_peak_mib": max(used, default=None),
            "ram_start_mib": memory[0] if memory else None, "ram_peak_mib": max(memory, default=None),
        })


def rank(row: dict) -> tuple:
    return (row["target_met"], row["balance"], row["balanced_accuracy"], row["f2"], row["roc_auc"])


def seed_statistics(group: list[dict]) -> dict:
    return {
        "recipe": group[0]["recipe"], "architecture": group[0]["architecture"], "seeds": len(group),
        "seeds_meeting_target": sum(row["target_met"] for row in group),
        **{f"{prefix}_{key}": value for key in METRICS for prefix, value in (
            ("mean", statistics.mean(row[key] for row in group)), ("sd", statistics.stdev(row[key] for row in group)),
        )},
        "mean_inference_ms": statistics.mean(row["inference_batch1_ms_median"] for row in group),
    }


def group_rank(group: dict) -> tuple:
    return (group["seeds_meeting_target"], group["mean_balance"], group["mean_balanced_accuracy"],
            group["mean_f2"], group["mean_roc_auc"], -group["sd_balance"], -group["mean_inference_ms"])


def representative(group: list[dict]) -> dict:
    """Choose the most central seed using the two prespecified balance statistics."""
    median_min = statistics.median(row["balance"] for row in group)
    median_mean = statistics.median(row["balanced_accuracy"] for row in group)
    return min(group, key=lambda row: (
        abs(row["balance"] - median_min), abs(row["balanced_accuracy"] - median_mean), row["seed"],
    ))


class Campaign:
    def __init__(self, dataset_dir: Path, manifest: Path, output: Path):
        self.dataset_dir, self.manifest, self.output = dataset_dir, manifest, output
        self.plan = read_json(PLAN)
        self.frozen = read_json(CONFIG_PATH)
        self.digest = self.frozen["dataset"]["manifest_sha256"]
        audit = read_json(output / "audit.json")
        if sha256_file(manifest) != self.digest or audit["splits"] != self.frozen["dataset"]["splits"]:
            raise ValueError("Ejecuta la auditoría contra la cohorte congelada.")
        for split, rows in _read_manifest(manifest).items():
            if split != "test" and cohort_identity(rows, dataset_dir) != audit["splits"][split]:
                raise ValueError(f"Inventario cambiado: {split}")
        self.validation = _read_manifest(manifest)["val"]
        self.cache = {}
        self.results = []
        self.attempted = 0
        self.smi = select_device(enable_amp=False)
        if self.smi.device.type != "cuda":
            raise RuntimeError("La campaña requiere CUDA.")
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"

    def train(self, phase: str, recipe: str, config: TrainConfig, precision: str = "fp32") -> dict | None:
        destination = self.output / phase / f"{recipe}-s{config.seed}"
        if destination.exists():
            requested = read_json(destination / "requested.json")
            if requested != {"configuration": asdict(config), "precision": precision}:
                raise ValueError(f"Configuración distinta en {destination}.")
            if not (destination / "record.json").exists():
                raise RuntimeError(f"Intento incompleto preservado en {destination}.")
            row = read_json(destination / "record.json")
            self.results.append(row)
            self.attempted += 1
            return row if row["status"] == "complete" else None
        if self.attempted >= self.plan["budget"]:
            raise RuntimeError("Se alcanzó el presupuesto de entrenamientos.")
        self.attempted += 1
        destination.mkdir(parents=True)
        _json(destination / "requested.json", {"configuration": asdict(config), "precision": precision})
        device = select_device(enable_amp=precision != "fp32",
                               amp_dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
        torch.cuda.empty_cache()
        row = {"phase": phase, "recipe": recipe, "seed": config.seed, "architecture": config.architecture,
               "training_precision": precision, **{f"cfg_{k}": v for k, v in asdict(config).items()}}
        print(f"[{self.attempted}/{self.plan['budget']}] {phase}/{recipe} semilla {config.seed}", flush=True)
        try:
            with telemetry(destination), (destination / "train.log").open("x", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log):
                    summary = run_training(self.dataset_dir, self.manifest, destination / "fit", config, device,
                                           shared_cache=self.cache)
        except (torch.OutOfMemoryError, FloatingPointError) as error:
            row.update(status="failed", failure=str(error))
            print(f"  Falló: {error}", flush=True)
        else:
            resources = read_json(destination / "resources.json")
            with (destination / "fit" / "history.csv").open(encoding="utf-8", newline="") as stream:
                times = [float(entry["epoch_seconds"]) for entry in csv.DictReader(stream)]
            spec = checkpoint_spec(destination / "fit" / "best.pt")
            metrics, _ = evaluate_model(
                self.dataset_dir, self.validation, self.digest, spec, destination / "validation",
                self.smi, "val", self.plan["target"], cached_pixels=self.cache.setdefault("val", {}),
            )
            row.update(metrics, **resources, status="complete", best_epoch=summary["best_epoch"],
                       training_seconds=summary["seconds"], epoch_seconds_mean=statistics.mean(times),
                       epoch_seconds_first=times[0], effective_batch=summary["effective_batch_size"],
                       **{key: summary[key] for key in (
                           "epochs_run", "peak_cuda_allocated_mib", "peak_cuda_reserved_mib",
                           "total_parameters", "trainable_parameters", "amp_skipped_steps",
                       )})
            print(f"  sens={row['recall']:.3f} esp={row['specificity']:.3f} "
                  f"AUC={row['roc_auc']:.3f} ({row['training_seconds']:.1f}s)", flush=True)
        _json(destination / "record.json", row)
        self.results.append(row)
        self.export()
        return row if row["status"] == "complete" else None

    def export(self):
        write_csv(self.output / "experiments.csv", self.results)
        _json(self.output / "experiments.json", {"attempted": self.attempted, "budget": self.plan["budget"],
                                                "runs": self.results})

    def run(self):
        saved_plan = self.output / "plan.json"
        if saved_plan.exists() and read_json(saved_plan) != self.plan:
            raise ValueError("El plan guardado es distinto.")
        if not saved_plan.exists():
            _json(saved_plan, self.plan)
            _json(self.output / "historical-audit.json", verify_history(self.manifest, self.frozen))
            _json(self.output / "environment.json", dependency_versions())
        seed, *others = self.plan["seeds"]
        controls = {}
        configurations = {}

        def sweep(phase, prefix, base, variants):
            rows = []
            for name, change in variants.items():
                recipe = f"{prefix}-{name}"
                config = replace(base, **change)
                configurations[recipe] = config
                row = self.train(phase, recipe, config)
                if row is not None:
                    rows.append(row)
            return rows

        for arch in self.plan["architectures"]:
            base = TrainConfig(architecture=arch, seed=seed, **self.plan["common"])
            controls[arch] = max(sweep("architecture", arch, base, self.plan["tuning"]), key=rank)
        finalists = sorted(controls, key=lambda arch: rank(controls[arch]), reverse=True)[:2]
        _json(self.output / "architecture-selection.json", {"advancing": finalists, "best_by_architecture": controls})
        for arch in finalists:
            anchor = controls[arch]
            best_loss = max([anchor, *sweep("loss", arch, configurations[anchor["recipe"]], self.plan["losses"])], key=rank)
            base = configurations[best_loss["recipe"]]
            variants = {key: value for key, value in self.plan["hyperparameters"].items()
                        if key != "lr-groups" or base.tuned_stages}
            sweep("hyperparameters", arch, base, variants)

        def quality_rows():
            return [row for row in self.results if row["status"] == "complete" and row["phase"] != "hardware"]

        anchor = max(quality_rows(), key=rank)
        base = configurations[anchor["recipe"]]
        if base.focal_loss:
            alternatives = (row for row in quality_rows() if row["architecture"] == base.architecture and not row["cfg_focal_loss"])
            bce_anchor = max(alternatives, key=rank)
            sweep("regularization", bce_anchor["architecture"], configurations[bce_anchor["recipe"]],
                  {"smoothing": self.plan["regularization"]["smoothing"]})
        sweep("regularization", base.architecture, base, {
            key: change for key, change in self.plan["regularization"].items() if key != "smoothing" or not base.focal_loss
        })
        shortlisted = sorted(quality_rows(), key=rank, reverse=True)[:3]
        _json(self.output / "shortlist.json", {"rows": shortlisted})
        complete = []
        seed_rows = []
        for anchor in shortlisted:
            recipe = anchor["recipe"]
            group = [anchor]
            for other in others:
                row = self.train("confirmation", recipe, replace(configurations[recipe], seed=other))
                if row is not None:
                    group.append(row)
            seed_rows.extend(group)
            if len(group) == len(self.plan["seeds"]):
                complete.append(group)
        write_csv(self.output / "seed_comparison.csv", seed_rows)
        ordered = sorted(complete, key=lambda group: group_rank(seed_statistics(group)), reverse=True)
        if not ordered:
            raise RuntimeError("Ninguna configuración finalista completó las tres semillas.")
        chosen = representative(ordered[0])
        decision = {"winner": chosen, "groups": [seed_statistics(group) for group in ordered],
                    "representatives": [representative(group) for group in ordered],
                    "best_by_architecture": controls, "manifest_sha256": self.digest}
        _json(self.output / "validation-selection.json", decision)
        write_csv(self.output / "seed_summary.csv", decision["groups"])
        base = configurations[chosen["recipe"]]
        for precision in ("fp32", "fp16", "bf16"):
            self.train("hardware", precision, replace(base, epochs=self.plan["hardware_epochs"]), precision)
        self.export()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    Campaign(args.dataset_dir, args.manifest, args.campaign).run()


if __name__ == "__main__":
    main()
