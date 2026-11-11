"""Run one phase of the frozen PAPILA research campaign, with per-run telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path

import torch

from src.device import select_device
from src.internal_test import sha256_file
from src.train import TrainConfig, run_training


PLAN_PATH = Path("configs/papila-research.json")


def phase_runs(plan: dict, phase: str, campaign: Path):
    if phase == "architectures":
        for seed in plan["seeds"]:
            for architecture in plan["architectures"]:
                yield architecture, architecture, seed, {}
    elif phase == "pilots":
        for name, changes in plan["pilots"].items():
            yield name, "resnet18", plan["seeds"][0], {"epochs": plan["pilot_epochs"], **changes}
    else:
        decision_phase = "architectures" if phase == "screen" else "screen"
        selection = json.loads((campaign / f"{decision_phase}-validation" / "selection.json").read_text("utf-8"))
        architecture = selection["architecture"]
        variants = plan["screen"] if phase == "screen" else selection["confirm_variants"]
        seeds = plan["seeds"][:1] if phase == "screen" else plan["seeds"][1:]
        for seed in seeds:
            for name in variants:
                yield name, architecture, seed, plan["screen"][name]


def save_telemetry(run_dir: Path, pending: Path) -> None:
    raw = run_dir / "gpu.csv"
    if pending.is_file():
        pending.rename(raw)
    rows = list(csv.reader(raw.read_text("utf-8").splitlines()))
    utilization = [float(row[1]) for row in rows if len(row) == 4 and row[1].strip().isdigit()]
    summary = {
        "samples": len(utilization),
        "mean_gpu_utilization_pct": statistics.mean(utilization) if utilization else None,
        "peak_gpu_utilization_pct": max(utilization, default=None),
    }
    (run_dir / "gpu-summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")


def train_phase(dataset_dir: Path, manifest: Path, campaign: Path, phase: str, *, cpu: bool) -> None:
    plan = json.loads(PLAN_PATH.read_text("utf-8"))
    if sha256_file(manifest) != plan["manifest_sha256"]:
        raise ValueError("El manifiesto difiere del protocolo congelado.")
    campaign.mkdir(parents=True, exist_ok=True)
    saved_plan = campaign / "plan.json"
    if saved_plan.exists() and json.loads(saved_plan.read_text("utf-8")) != plan:
        raise ValueError("El plan difiere del guardado en esta campaña.")
    if not saved_plan.exists():
        shutil.copyfile(PLAN_PATH, saved_plan)
    parent = campaign / phase
    parent.mkdir(exist_ok=True)
    smi = shutil.which("nvidia-smi")
    for name, architecture, seed, changes in phase_runs(plan, phase, campaign):
        run_dir = parent / f"{name}-s{seed}"
        telemetry = parent / f"{run_dir.name}-gpu.csv"
        if (run_dir / "summary.json").is_file():
            if not (run_dir / "gpu-summary.json").is_file():
                save_telemetry(run_dir, telemetry)
            continue
        options = {**plan["common"], **changes}
        precision = options.pop("precision", "fp32")
        pinned = options.pop("pin_memory", True)
        device = select_device(
            prefer_cuda=not cpu, enable_amp=precision != "fp32",
            amp_dtype=torch.bfloat16 if precision == "bf16" else torch.float16, pin_memory=pinned,
        )
        if device.device.type != "cuda" and not cpu:
            raise RuntimeError("CUDA no disponible. Usa --cpu para una campaña explícitamente en CPU.")
        if cpu and phase == "pilots":
            raise ValueError("Los pilotos de recursos requieren CUDA.")
        if device.device.type == "cuda":
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        config = TrainConfig(**options, architecture=architecture, seed=seed)
        print(f"=== {run_dir} · {device.device} · {precision} ===", flush=True)
        if run_dir.exists():
            raise FileExistsError(f"Conserva la corrida incompleta en otra carpeta antes de reintentar: {run_dir}")
        try:
            with telemetry.open("x", encoding="utf-8") as stream:
                monitor = None
                if smi and not cpu:
                    monitor = subprocess.Popen([
                        smi, "--id=0", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits", "--loop-ms=500",
                    ], stdout=stream, stderr=subprocess.DEVNULL)
                try:
                    summary = run_training(dataset_dir, manifest, run_dir, config, device)
                finally:
                    if monitor is not None and monitor.poll() is None:
                        monitor.terminate()
                        monitor.wait(timeout=10)
        finally:
            if run_dir.is_dir() and telemetry.is_file():
                save_telemetry(run_dir, telemetry)
        print(f"Completada {run_dir.name}, época {summary['best_epoch']}, {summary['seconds']:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("architectures", "screen", "confirm", "pilots"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    train_phase(args.dataset_dir, args.manifest, args.campaign, args.phase, cpu=args.cpu)


if __name__ == "__main__":
    main()
