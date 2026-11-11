"""Check both frozen candidates through the GlaucoAI routes in CPU and CUDA."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import FRONTEND_DIST, create_app, load_service
from src.data import _read_manifest
from src.internal_test import sha256_file


def verify(spec: dict, samples: dict[str, bytes], expected: dict[str, float], *, use_cuda: bool) -> dict:
    load = partial(load_service, Path(spec["checkpoint"]), spec["sha256"], spec["threshold"], prefer_cuda=use_cuda)
    frontend = FRONTEND_DIST if (FRONTEND_DIST / "index.html").is_file() else None
    app = create_app(frontend, model_loader=load)
    headers = {"Content-Type": "image/jpeg"}
    timings = {"predict": [], "analyze": []}
    observed = {}
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        if frontend is not None:
            client.get("/").raise_for_status()
        health = client.get("/api/health").raise_for_status().json()
        device = torch.device(health["device"])
        if (device.type == "cuda") != use_cuda or health["threshold"] != spec["threshold"]:
            raise RuntimeError(f"Dispositivo o umbral inesperado: {health}")
        for image_id, payload in samples.items():
            predict = client.post("/api/predict", content=payload, headers=headers).raise_for_status().json()
            analyze = client.post("/api/analyze", content=payload, headers=headers).raise_for_status().json()
            explain = client.post("/api/explain", content=payload, headers=headers).raise_for_status().json()
            for field in ("heatmap_png", "overlay_png"):
                if analyze[field] != explain[field]:
                    raise ValueError(f"Grad-CAM difiere entre /analyze y /explain: {field}")
                with Image.open(io.BytesIO(base64.b64decode(analyze[field], validate=True))) as png:
                    if png.format != "PNG":
                        raise ValueError("Se esperaba un PNG.")
                    png.verify()
            score = predict["score_glaucoma"]
            if not math.isclose(score, analyze["score_glaucoma"], abs_tol=1e-6):
                raise ValueError("Los puntajes de /predict y /analyze difieren.")
            if not math.isclose(score, expected[image_id], abs_tol=1e-4):
                raise ValueError("La API y la evaluación de validación difieren.")
            if not (predict["label"] == analyze["label"] == int(score >= spec["threshold"])):
                raise ValueError("La predicción no aplica el umbral fijado.")
            observed[image_id] = {"score": score, "label": predict["label"],
                                  "target_layer": analyze["target_layer"], "has_signal": analyze["has_signal"]}
        for route, durations in timings.items():
            for _ in range(3):
                for payload in samples.values():
                    start = time.perf_counter()
                    client.post(f"/api/{route}", content=payload, headers=headers).raise_for_status()
                    durations.append((time.perf_counter() - start) * 1e3)
    return {"health": health, "frontend_served": frontend is not None, "images": observed,
            "mean_ms": {route: statistics.mean(samples) for route, samples in timings.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    lock = json.loads((args.campaign / "test-lock.json").read_text("utf-8"))
    if sha256_file(args.manifest) != lock["manifest_sha256"]:
        raise ValueError("El manifiesto difiere del fijado.")
    val = _read_manifest(args.manifest)["val"]
    records = [next(item for item in val if item.label == label) for label in (0, 1)]
    samples = {item.image_id: (args.dataset_dir / item.relative_path).read_bytes() for item in records}
    output = args.campaign / "backend-verification.json"
    if output.exists():
        raise FileExistsError(output)
    if not args.cpu_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible. Usa --cpu-only para verificar solo CPU.")
    report = {}
    for name in ("demo", "candidate"):
        spec = lock["models"][name]
        path = args.campaign / "locked-validation" / name / "predictions.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            expected = {row["image_id"]: float(row["score"]) for row in csv.DictReader(stream)}
        report[name] = {"cpu": verify(spec, samples, expected, use_cuda=False)}
        if not args.cpu_only:
            report[name]["cuda"] = verify(spec, samples, expected, use_cuda=True)
            for image_id in samples:
                cpu, cuda = (report[name][device]["images"][image_id] for device in ("cpu", "cuda"))
                if cpu["label"] != cuda["label"] or not math.isclose(cpu["score"], cuda["score"], abs_tol=1e-4):
                    raise ValueError("La predicción difiere entre CPU y CUDA.")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
