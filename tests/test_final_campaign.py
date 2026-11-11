"""Check partial accumulation, checkpoint compatibility and clustered AUC."""

import io
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_final import bootstrap_values, weight_draws
from scripts.train_final import telemetry
from src.device import select_device
from src.model import CHECKPOINT_VERSION, CLASSES, load_checkpoint
from src.pretrained import RESNET18, PapilaResNet18, imagenet_normalization
from src.train import _train_epoch


class FinalCampaignTests(unittest.TestCase):
    def test_resources_are_preserved_if_training_fails(self):
        with TemporaryDirectory() as directory, patch("scripts.train_final.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "fallo"):
                with telemetry(Path(directory)):
                    raise RuntimeError("fallo")
            self.assertEqual(json.loads((Path(directory) / "resources.json").read_text())["gpu_samples"], 0)

    def test_accumulation_uses_the_partial_group_size(self):
        dataset = TensorDataset(torch.arange(7).float().unsqueeze(1) / 7, torch.tensor([0., 1., 1., 0., 0., 1., 1.]))
        device = select_device(prefer_cuda=False, enable_amp=False)

        class BinaryLinear(nn.Linear):
            def forward(self, inputs):
                return super().forward(inputs).squeeze(1)

        large = BinaryLinear(1, 1)
        micro = BinaryLinear(1, 1)
        micro.load_state_dict(large.state_dict())
        for model, batch, steps in ((large, 4, 1), (micro, 2, 2)):
            _train_epoch(model, DataLoader(dataset, batch_size=batch), device, nn.BCEWithLogitsLoss(),
                         torch.optim.SGD(model.parameters(), lr=0.1), device.make_grad_scaler(), steps)
        torch.testing.assert_close(large.weight, micro.weight)
        torch.testing.assert_close(large.bias, micro.bias)

    def test_dropout_checkpoint_round_trip(self):
        model = PapilaResNet18(pretrained=False, dropout=0.2)
        self.assertTrue(model.classification_head[0].training)
        buffer = io.BytesIO()
        torch.save({
            "format_version": CHECKPOINT_VERSION, "classes": CLASSES,
            "architecture": {"name": RESNET18, "image_size": 224, "dropout": 0.2},
            "normalization": asdict(imagenet_normalization()), "state_dict": model.state_dict(),
        }, buffer)
        buffer.seek(0)
        restored, _ = load_checkpoint(buffer)
        self.assertFalse(restored.classification_head[0].training)
        torch.testing.assert_close(model.fc_out.weight, restored.fc_out.weight)

    def test_bootstrap_preserves_patient_and_counts_score_ties(self):
        draws = weight_draws(["A", "A", "B", "B"], replicates=25, seed=42)
        np.testing.assert_equal(draws[:, 0], draws[:, 1])
        np.testing.assert_equal(draws[:, 2], draws[:, 3])
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.4, 0.8])
        weights = np.array([[1., 1., 2., 2.], [0., 0., 1., 1.]])
        result = bootstrap_values(labels, scores, scores >= 0.4, weights)
        for i, draw in enumerate(weights):
            self.assertAlmostEqual(result["roc_auc"][i], roc_auc_score(labels, scores, sample_weight=draw))
            self.assertAlmostEqual(result["average_precision"][i], average_precision_score(labels, scores, sample_weight=draw))
