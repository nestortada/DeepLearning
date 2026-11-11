"""Benchmark the compiled app with local images from the frozen validation split."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from backend import main
from src.data import _read_manifest


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _timed(function, times):
    def call(*args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        times.append((perf_counter() - start) * 1000)
        return result
    return call


def _gpu_memory(device):
    torch.cuda.synchronize(device)
    return {
        "allocated": round(torch.cuda.memory_allocated(device) / 2**20, 2),
        "reserved": round(torch.cuda.memory_reserved(device) / 2**20, 2),
        "peak_allocated": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "peak_reserved": round(torch.cuda.max_memory_reserved(device) / 2**20, 2),
    }


def benchmark(payloads, samples):
    app = main.create_app(main.FRONTEND_DIST)
    load, inference, grad_cam = [], [], []
    with (
        patch.object(main, "load_model", _timed(main.load_model, load)),
        patch.object(main, "score_image", _timed(main.score_image, inference)),
        patch.object(main, "explain_image", _timed(main.explain_image, grad_cam)),
        TestClient(app, base_url="http://127.0.0.1:8000") as client,
    ):
        service = app.state.service
        device = service.device
        is_cuda = device.type == "cuda"

        def send(route, payload):
            start = perf_counter()
            response = client.post(f"/api/{route}", content=payload, headers={"Content-Type": "image/jpeg"})
            response.raise_for_status()
            return (perf_counter() - start) * 1000

        for payload in payloads:
            send("predict", payload)
            send("analyze", payload)
        inference.clear()
        grad_cam.clear()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        warm = _gpu_memory(device) if is_cuda else None

        batch = [payloads[index % len(payloads)] for index in range(samples)]
        predict_ms = [send("predict", payload) for payload in batch]
        analyze_ms = [send("analyze", payload) for payload in batch]
        sequential_cam_ms = mean(grad_cam)
        after_sequential = _gpu_memory(device) if is_cuda else None

        start = perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_ms = list(pool.map(lambda payload: send("analyze", payload), batch))
        concurrent_total_ms = (perf_counter() - start) * 1000
        after_concurrent = _gpu_memory(device) if is_cuda else None
        if service.model.gradcam_layer._forward_hooks:
            raise RuntimeError("Quedaron hooks activos tras las solicitudes.")

    return {
        "model": service.model.architecture, "threshold": service.threshold,
        "device": str(device), "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(device) if is_cuda else None,
        "samples_per_phase": samples, "cold_model_load_ms": round(load[0], 2),
        "mean_fp32_inference_ms": round(mean(inference), 2),
        "mean_fp32_grad_cam_ms": round(sequential_cam_ms, 2),
        "mean_predict_response_ms": round(mean(predict_ms), 2),
        "mean_analyze_response_ms": round(mean(analyze_ms), 2),
        "mean_analyze_response_ms_two_clients": round(mean(concurrent_ms), 2),
        "concurrent_batch_ms": round(concurrent_total_ms, 2),
        "gpu_memory_mib": {"warm": warm, "after_sequential": after_sequential,
                           "after_concurrent": after_concurrent} if is_cuda else None,
    }


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-id", action="append", required=True, help="ID de validación. Se puede repetir.")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("Usa al menos dos solicitudes por fase.")
    if args.cpu:
        os.environ["GLAUCOAI_CPU"] = "1"
    try:
        config = json.loads((main.ROOT / "configs" / "papila-internal-test.json").read_text(encoding="utf-8"))
        manifest_sha = config["dataset"]["manifest_sha256"]
        if sha256(args.manifest) != manifest_sha:
            raise ValueError("El manifiesto no coincide con el congelado.")
        validation = {row.image_id: row for row in _read_manifest(args.manifest)["val"]}
        if not set(args.image_id).issubset(validation):
            raise ValueError("Selecciona exclusivamente IDs de la partición de validación.")
        payloads = [(args.dataset_dir / validation[image_id].relative_path).read_bytes() for image_id in args.image_id]
        spec = main.model_spec()
        report = benchmark(payloads, args.samples)
        if sha256(spec.checkpoint) != spec.sha256 or sha256(args.manifest) != manifest_sha:
            raise ValueError("El checkpoint o el manifiesto cambió durante la medición.")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
