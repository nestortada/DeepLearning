"""Exercise frozen ResNet18, letterbox preprocessing and validation-only comparison."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.compare import analyze_thresholds, compare_validation, select_threshold
from src.data import make_image_transform
from src.device import select_device
from src.model import load_checkpoint
from src.pretrained import RESNET18, PapilaResNet18, imagenet_normalization
from src.train import TrainConfig, run_training, validation_metrics


class TransferLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled())
        torch.set_num_threads(1)

    def test_letterbox_preserves_horizontal_image_and_uses_imagenet_stats(self) -> None:
        with Image.new("RGB", (80, 40), (255, 0, 0)) as image:
            pixels = make_image_transform(224, preserve_aspect=True)(image)
        self.assertEqual(pixels.shape, (3, 224, 224))
        self.assertTrue((pixels[:, :56] == 0).all().item())
        self.assertTrue((pixels[0, 56:168] == 1).all().item())
        self.assertTrue((pixels[:, 168:] == 0).all().item())
        self.assertEqual(imagenet_normalization().mean, (0.485, 0.456, 0.406))

    def _verify_head_update(self, *, cuda: bool) -> None:
        device = select_device(prefer_cuda=cuda)
        self.assertEqual(device.device.type, "cuda" if cuda else "cpu")
        torch.manual_seed(42)
        model = PapilaResNet18(pretrained=False).to(device.device).train()
        self.assertFalse(model.backbone.bn1.training)
        self.assertTrue(model.fc_out.training)
        self.assertEqual([name for name, param in model.named_parameters() if param.requires_grad],
                         ["backbone.fc.weight", "backbone.fc.bias"])
        frozen = {name: tensor.clone() for name, tensor in model.state_dict().items()
                  if not name.startswith("backbone.fc.")}
        head = model.fc_out.weight.detach().clone()
        optimizer = torch.optim.Adam(model.fc_out.parameters(), lr=0.001)
        scaler = device.make_grad_scaler()
        with device.autocast():
            logits = model(torch.full((2, 3, 32, 32), 0.1, device=device.device))
            self.assertEqual(logits.shape, (2,))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits).float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        self.assertFalse(torch.equal(head, model.fc_out.weight))
        self.assertTrue(all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in frozen.items()))
        self.assertIsNone(model.backbone.conv1.weight.grad)

    def test_cpu_head_updates_and_backbone_stays_frozen(self) -> None:
        self._verify_head_update(cuda=False)

    def test_last_block_tuning_updates_only_convolutions_and_head(self) -> None:
        model = PapilaResNet18(pretrained=False, tune_last_block=True).train()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        model(torch.randn(2, 3, 32, 32)).sum().backward()
        optimizer.step()
        changed = {name for name, tensor in model.state_dict().items()
                   if not torch.equal(before[name], tensor)}
        self.assertIn("backbone.layer4.0.conv1.weight", changed)
        self.assertIn("backbone.fc.weight", changed)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.assertFalse(module.training, name)
                self.assertTrue(all(not p.requires_grad for p in module.parameters()), name)
        self.assertTrue(all(
            name.startswith("backbone.fc.") or name.startswith("backbone.layer4.")
            for name in changed
        ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA no disponible")
    def test_cuda_amp_head_updates_and_backbone_stays_frozen(self) -> None:
        self._verify_head_update(cuda=True)

    def test_threshold_selection_and_ap(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        scores = torch.tensor([0.20, 0.55, 0.45, 0.65])
        rows = analyze_thresholds(labels, scores)
        self.assertEqual(len(rows), 21)
        self.assertEqual(select_threshold(rows)["threshold"], 0.65)
        metrics = validation_metrics(labels, scores.logit(), 0.5)
        self.assertAlmostEqual(metrics["val_average_precision"], 5 / 6)
        self.assertNotEqual(metrics["val_average_precision"], metrics["val_pr_auc"])
        self.assertEqual(metrics["val_specificity"], 0.5)
        self.assertEqual(metrics["val_f2"], 0.5)

    def test_reproducible_checkpoint_and_comparison_without_test_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("image_id", "patient_id", "eye", "label", "relative_path", "split"))
                for patient, split in enumerate(("train", "val", "test"), start=1):
                    for eye, label in (("OD", 0), ("OS", 1)):
                        name = f"RET{patient:03d}{eye}"
                        writer.writerow((name, patient, eye, label, f"{name}.png", split))
                        if split != "test":
                            with Image.new("RGB", (32, 16), (30 + 100 * label, 40, 60)) as image:
                                image.save(root / f"{name}.png")
            cpu = select_device(prefer_cuda=False)
            config = TrainConfig(architecture=RESNET18, image_size=32, epochs=1, batch_size=2)
            with patch("src.train.PapilaResNet18", side_effect=lambda **_: PapilaResNet18(pretrained=False)), \
                    redirect_stdout(io.StringIO()):
                first = run_training(root, manifest, root / "resnet", config, cpu)
                second = run_training(root, manifest, root / "repeat", config, cpu)
                run_training(root, manifest, root / "cnn", TrainConfig(image_size=8, epochs=1, batch_size=2), cpu)
            self.assertEqual(first["validation"], second["validation"])
            self.assertEqual(first["trainable_parameters"], 513)
            self.assertTrue(first["updated_head"])
            model, payload = load_checkpoint(root / "resnet" / "best.pt")
            repeated, _ = load_checkpoint(root / "repeat" / "best.pt")
            self.assertTrue(torch.equal(model.fc_out.weight, repeated.fc_out.weight))
            self.assertEqual(payload["preprocessing"]["geometry"], "letterbox")
            self.assertFalse(model.training)
            report = compare_validation(root, manifest, root / "cnn", root / "resnet", root / "comparison", cpu=True)
            self.assertEqual((report["images"], report["positives"]), (2, 1))
            self.assertEqual(report["models"]["ResNet18"]["validation_at_0_5"], first["validation"])
            self.assertTrue((root / "comparison" / "curves.png").is_file())
