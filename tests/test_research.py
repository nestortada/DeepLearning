"""Protect threshold selection and the new checkpoint and Grad-CAM paths."""

import io
import unittest
from dataclasses import asdict

import torch
from PIL import Image
from torch import nn

from scripts.evaluate_research import threshold_sweep
from src.explain import explain_image
from src.model import CHECKPOINT_VERSION, CLASSES, load_checkpoint
from src.pretrained import EFFICIENTNET_B0, RESNET34, PapilaTransferModel, imagenet_normalization


class ResearchThresholdTests(unittest.TestCase):
    def test_threshold_examines_midpoints_and_chooses_balance(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        scores = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.04, 0.05])
        selected, rows = threshold_sweep(labels, scores, 0.85)
        self.assertAlmostEqual(selected["threshold"], 0.035)
        self.assertEqual((selected["tn"], selected["tp"]), (3, 3))
        self.assertEqual(len(rows), 6)

    def test_threshold_falls_back_to_balanced_point_when_target_is_unattainable(self):
        selected, _ = threshold_sweep(
            torch.tensor([0, 1, 0, 1]), torch.tensor([0.1, 0.2, 0.3, 0.4]), 0.85,
        )
        self.assertEqual((selected["recall"], selected["specificity"]), (1.0, 0.5))


class ResearchCheckpointTests(unittest.TestCase):
    def test_new_checkpoints_reload_and_explain(self):
        for architecture in (RESNET34, EFFICIENTNET_B0):
            with self.subTest(architecture=architecture):
                trained = PapilaTransferModel(architecture, pretrained=False, tune_blocks=1, freeze_batch_norm=False)
                trained.configure_tuning(enabled=False)
                self.assertFalse(any(m.training for m in trained.modules() if isinstance(m, nn.BatchNorm2d)))
                trained.configure_tuning(enabled=True)
                self.assertTrue(any(m.training for m in trained.modules() if isinstance(m, nn.BatchNorm2d)))
                stream = io.BytesIO()
                torch.save({
                    "format_version": CHECKPOINT_VERSION, "classes": CLASSES,
                    "architecture": {"name": architecture, "image_size": 64},
                    "preprocessing": {"geometry": "letterbox"},
                    "normalization": asdict(imagenet_normalization(architecture)),
                    "state_dict": trained.state_dict(),
                }, stream)
                stream.seek(0)
                model, checkpoint = load_checkpoint(stream)
                self.assertFalse(any(m.training for m in model.modules()))
                with Image.new("RGB", (64, 48), (20, 40, 60)) as image:
                    result = explain_image(model, image, checkpoint, threshold=0.4)
                self.assertEqual(tuple(result.heatmap.shape), (48, 64))
                self.assertTrue(result.heatmap.isfinite().all())
                self.assertFalse(model.gradcam_layer._forward_hooks)
