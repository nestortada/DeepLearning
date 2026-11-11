"""Compare saved eye-level errors and Grad-CAM maps on the locked PAPILA cohorts."""

from __future__ import annotations

import argparse
import csv
import io
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageOps

from src.data import _read_manifest
from src.device import select_device
from src.explain import explain_image, summarize_regions
from src.explain_visualization import make_overlay
from src.internal_test import CONFIG_PATH, _json, cohort_identity, sha256_file
from src.model import load_checkpoint

from .evaluate_final import verify_lock
from .evaluate_research import read_json
from .train_final import write_csv


MODELS = ("candidate", "resnet18-control", "resnet34-control", "efficientnet_b0-control")


def quality(image: Image.Image) -> dict:
    rgb = np.asarray(image)
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
    visible = rgb.max(axis=2) > 20
    core = gray[1:-1, 1:-1]
    laplacian = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * core
    interior = (visible[1:-1, 1:-1] & visible[:-2, 1:-1] & visible[2:, 1:-1]
                & visible[1:-1, :-2] & visible[1:-1, 2:])
    return {
        "visible_mean_luma": float(gray[visible].mean()) if visible.any() else None,
        "visible_std_luma": float(gray[visible].std()) if visible.any() else None,
        "visible_laplacian_variance": float(laplacian[interior].var()) if interior.any() else None,
        "dark_fraction": float(1 - visible.mean()),
    }


def anatomical_regions(image_id: str, preview: Image.Image, original_size: tuple, contours: Path | None) -> dict:
    regions = {}
    width, height = preview.size
    if contours is not None:
        for name in ("disc_exp1", "disc_exp2", "cup_exp1", "cup_exp2"):
            path = contours / f"{image_id}_{name}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            coordinates = np.loadtxt(io.StringIO(path.read_text("utf-8-sig").replace(",", " ")))
            if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
                raise ValueError(f"Contorno inválido: {path}")
            pixels = coordinates * (width / original_size[0], height / original_size[1])
            with Image.new("1", preview.size) as mask:
                ImageDraw.Draw(mask).polygon([tuple(point) for point in pixels], fill=1)
                regions[name] = torch.from_numpy(np.asarray(mask).copy())
        for structure in ("disc", "cup"):
            regions[f"{structure}_intersection"] = regions[f"{structure}_exp1"] & regions[f"{structure}_exp2"]
            regions[f"{structure}_union"] = regions[f"{structure}_exp1"] | regions[f"{structure}_exp2"]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    regions["central_ellipse"] = ((x / width - 0.5) / 0.2).square() + ((y / height - 0.5) / 0.2).square() <= 1
    regions["outer_10_percent"] = (x < width * 0.1) | (x >= width * 0.9) | (y < height * 0.1) | (y >= height * 0.9)
    regions["dark_rgb_max_le_20"] = torch.from_numpy(np.asarray(preview).max(axis=2) <= 20)
    return regions


def category(label: int, predicted: int) -> str:
    return {(0, 0): "VN", (0, 1): "FP", (1, 0): "FN", (1, 1): "VP"}[label, predicted]


def contour_overlay(image: Image.Image, regions: dict) -> Image.Image:
    pixels = np.array(image)
    for region, color in (("disc_union", (0, 240, 255)), ("cup_union", (255, 255, 255))):
        if region in regions:
            mask = regions[region].numpy()
            inner = mask.copy()
            inner[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
            pixels[mask & ~inner] = color
    return Image.fromarray(pixels)


def save_sheet(output: Path, examples: list, keys: list[str], split: str, *, expert_contours: bool) -> None:
    figure = Figure(figsize=(3 * (len(keys) + 1), 2.15 * len(examples) + 0.6), layout="constrained")
    FigureCanvasAgg(figure)
    grid = figure.subplots(len(examples), len(keys) + 1, squeeze=False)
    legend = " · Disco cian, copa blanca (unión de expertos)" if expert_contours else ""
    figure.suptitle(f"PAPILA · {split} · Grad-CAM del logit de glaucoma{legend}")
    for axes, (row, preview, overlays) in zip(grid, examples, strict=True):
        axes[0].imshow(preview)
        axes[0].set_title(f"{row['image_id']} · {'glaucoma' if row['label'] else 'sano'}", fontsize=9)
        for axis, name in zip(axes[1:], keys, strict=True):
            axis.imshow(overlays[name])
            axis.set_title(f"{name} · {row[name + '_category']} · {row[name + '_score']:.3f}", fontsize=8)
        for axis in axes:
            axis.set_axis_off()
    figure.savefig(output, dpi=115)


def analyze_split(dataset_dir: Path, records: list, campaign: Path, report: dict, output: Path,
                  models: dict, split: str, contours: Path | None) -> dict:
    folder = campaign / ("locked-validation" if split == "val" else "test")
    keys = list(models)
    predictions = {}
    for key in keys:
        with (folder / key / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            predictions[key] = {row["image_id"]: row for row in csv.DictReader(stream)}
    output.mkdir()
    rows, cam_rows, panels = [], [], []
    groups = defaultdict(list)
    selected_controls = Counter()
    for item in sorted(records, key=lambda record: record.image_id):
        row = {"image_id": item.image_id, "patient_id": item.patient_id, "eye": item.eye, "label": item.label}
        for name in keys:
            prediction = predictions[name][item.image_id]
            if (prediction["patient_id"], int(prediction["label"])) != (item.patient_id, item.label):
                raise ValueError("Identidad del ojo distinta en las predicciones.")
            row[f"{name}_score"] = float(prediction["score"])
            row[f"{name}_category"] = category(item.label, int(prediction["prediction"]))
        disagreement = len({row[f"{key}_category"] for key in keys}) > 1
        candidate_category = row["candidate_category"]
        error = candidate_category in ("FP", "FN")
        control = not (error or disagreement) and selected_controls[item.label] < 3
        selected_controls[item.label] += int(control)
        row.update(disagreement=disagreement, selected_for_gradcam=error or disagreement or control)
        with Image.open(dataset_dir / item.relative_path) as source:
            image = source.convert("RGB")
        preview = ImageOps.contain(image, (384, 384), Image.Resampling.BILINEAR)
        row.update(quality(preview))
        rows.append(row)
        groups[item.patient_id].append(row)
        if not row["selected_for_gradcam"]:
            continue
        regions = anatomical_regions(item.image_id, preview, image.size, contours)
        overlays = {}
        for name, (model, checkpoint) in models.items():
            threshold = report["models"][name]["threshold"]
            cam = explain_image(model, image, checkpoint, threshold=threshold, output_size=preview.size)
            if abs(cam.score_glaucoma - row[f"{name}_score"]) > 1e-4:
                raise ValueError(f"Grad-CAM no reproduce el puntaje de {item.image_id}/{name}.")
            stats = summarize_regions(cam.heatmap, regions)
            cam_row = {
                "image_id": item.image_id, "patient_id": item.patient_id, "model": name,
                "category": row[f"{name}_category"], "score": cam.score_glaucoma,
                "grid_height": cam.feature_shape[0], "grid_width": cam.feature_shape[1],
                "has_signal": bool(cam.heatmap.any()),
            }
            for region, values in stats.items():
                cam_row[f"{region}_area"] = values["area_fraction"]
                cam_row[f"{region}_mass"] = values["cam_fraction"]
                cam_row[f"{region}_ratio"] = values["cam_to_area_ratio"]
            cam_rows.append(cam_row)
            overlays[name] = contour_overlay(make_overlay(preview, cam.heatmap, alpha=0.62), regions)
        panels.append((row, preview, overlays))
        if len(panels) == 6:
            save_sheet(output / f"gradcam-{len(cam_rows) // len(keys) - 5:03}.png", panels, keys, split,
                       expert_contours=contours is not None)
            panels.clear()
    if panels:
        save_sheet(output / f"gradcam-{len(cam_rows) // len(keys) - len(panels) + 1:03}.png", panels, keys, split,
                   expert_contours=contours is not None)

    patients = []
    for patient_id, eyes in sorted(groups.items()):
        errors = sum(eye["candidate_category"] in ("FN", "FP") for eye in eyes)
        patients.append({"patient_id": patient_id, "affected_eyes": errors, "eyes": len(eyes),
                         **{f"{eye['eye']}_{field}": eye[field] for eye in eyes for field in
                            ("image_id", "label", "candidate_category", "candidate_score")}})
    write_csv(output / "eyes.csv", rows)
    write_csv(output / "errors.csv", [row for row in rows if row["candidate_category"] in ("FP", "FN")])
    write_csv(output / "patients.csv", patients)
    write_csv(output / "gradcam.csv", cam_rows)
    metrics = {}
    for label in ("VN", "FP", "FN", "VP"):
        eyes = [row for row in rows if row["candidate_category"] == label]
        metrics[label] = {"eyes": len(eyes), "patients": len({row["patient_id"] for row in eyes})}
        for field in ("visible_mean_luma", "visible_std_luma", "visible_laplacian_variance", "dark_fraction"):
            measured = [row[field] for row in eyes if row[field] is not None]
            metrics[label][f"median_{field}"] = statistics.median(measured) if measured else None
        maps = [row for row in cam_rows if row["model"] == "candidate" and row["category"] == label]
        metrics[label]["gradcam_with_signal"] = sum(row["has_signal"] for row in maps)
        for region in ("central_ellipse", "outer_10_percent", "dark_rgb_max_le_20", "disc_union", "cup_union"):
            ratios = [row[f"{region}_ratio"] for row in maps if row.get(f"{region}_ratio") is not None]
            metrics[label][f"median_cam_{region}_ratio"] = statistics.median(ratios) if ratios else None
    return {
        "cohort_eyes": len(rows), "models": keys, "candidate_categories": metrics,
        "discordant_eyes": sum(row["disagreement"] for row in rows),
        "gradcam_eyes": len(cam_rows) // len(keys), "affected_patients": sum(row["affected_eyes"] > 0 for row in patients),
        "patients_with_both_eyes_misclassified": sum(row["affected_eyes"] == 2 for row in patients),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/papila"))
    parser.add_argument("--manifest", type=Path, default=Path("data/papila/manifest-c.csv"))
    parser.add_argument("--contours", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = select_device(enable_amp=False).device
    if device.type == "cuda":
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    report = verify_lock(args.manifest, args.campaign)
    if not (args.campaign / "metrics.json").is_file():
        raise ValueError("Completa el test bloqueado antes del análisis de errores.")
    output = args.campaign / "error-analysis"
    output.mkdir()
    models = {}
    for key in MODELS:
        if key not in report["models"]:
            continue
        spec = report["models"][key]
        checkpoint_path = Path(spec["checkpoint"])
        if sha256_file(checkpoint_path) != spec["sha256"]:
            raise ValueError(f"Checkpoint alterado: {checkpoint_path}")
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        model.requires_grad_(False)
        del checkpoint["state_dict"]
        models[key] = model, checkpoint
    frozen = read_json(CONFIG_PATH)["dataset"]["splits"]
    results = {}
    records = _read_manifest(args.manifest)
    for split in ("val", "test"):
        if cohort_identity(records[split], args.dataset_dir) != frozen[split]:
            raise ValueError(f"Inventario alterado: {split}.")
        results[split] = analyze_split(args.dataset_dir, records[split], args.campaign, report,
                                       output / split, models, split, args.contours)
    _json(output / "summary.json", {
        "selection": "Todos los errores del candidato, desacuerdos entre modelos y tres controles por clase",
        "expert_contours": str(args.contours) if args.contours else None,
        "device": str(device), **results,
    })


if __name__ == "__main__":
    main()
