"""Exercise the loaders with a small synthetic manifest and images."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image
from torchvision import transforms

from src.data import make_loaders
from src.device import select_device


class PapilaLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.cpu = select_device(prefer_cuda=False)
        self.rows: list[dict[str, str]] = []
        images = self.root / "FundusImages"
        images.mkdir()
        for patient in range(1, 11):
            split = "train" if patient < 9 else "val" if patient == 9 else "test"
            for eye, label in (("OD", 0), ("OS", 1)):
                image_id = f"RET{patient:03d}{eye}"
                path = images / f"{image_id}.png"
                with Image.new("RGB", (4, 4)) as image:
                    image.paste((255 if split != "train" else 0, 64, label * 255), (0, 0, 4, 4))
                    image.paste((255, 64, label * 255), (2, 0, 4, 4))
                    image.save(path)
                self.rows.append({
                    "image_id": image_id,
                    "patient_id": f"{patient:03d}",
                    "eye": eye,
                    "label": str(label),
                    "relative_path": path.relative_to(self.root).as_posix(),
                    "split": split,
                })
        self._write_manifest()

    def _write_manifest(self) -> None:
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_loads_train_batch_with_expected_shape_and_labels(self) -> None:
        loaders = make_loaders(self.root, self.manifest, device=self.cpu, augment_train=False)
        images, labels = next(iter(loaders.train))
        self.assertEqual(images.shape, (16, 3, 96, 96))
        self.assertEqual(labels.shape, (16,))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.float32)
        self.assertTrue(torch.isfinite(images).all().item())
        torch.testing.assert_close(labels, (images[:, 2, 0, 0] > 0).float())
        self.assertEqual(loaders.train.num_workers, 0)
        self.assertFalse(loaders.train.pin_memory)

    def test_normalization_uses_train_only_and_handles_constant_channel(self) -> None:
        loaders = make_loaders(
            self.root, self.manifest, device=self.cpu, image_size=4, augment_train=False,
        )
        torch.testing.assert_close(torch.tensor(loaders.normalization.mean), torch.tensor([0.5, 64 / 255, 0.5]))
        self.assertEqual(loaders.normalization.std, (0.5, 1.0, 0.5))
        train, _ = next(iter(loaders.train))
        val, _ = next(iter(loaders.val))
        torch.testing.assert_close(train.mean(dim=(0, 2, 3)), torch.zeros(3))
        torch.testing.assert_close(val[:, 0], torch.ones_like(val[:, 0]))

    def test_respects_manifest_and_has_reproducible_train_order(self) -> None:
        original = self.manifest.read_bytes()
        first = make_loaders(self.root, self.manifest, device=self.cpu, batch_size=3, include_test=True)
        second = make_loaders(self.root, self.manifest, device=self.cpu, batch_size=3)
        order = list(first.train.sampler)
        self.assertEqual(order, list(second.train.sampler))
        self.assertNotEqual(order, list(range(16)))
        self.assertEqual(self.manifest.read_bytes(), original)
        for split, loader in (("train", first.train), ("val", first.val), ("test", first.test)):
            self.assertIsNotNone(loader)
            expected = [row for row in self.rows if row["split"] == split]
            self.assertEqual([record.image_id for record in loader.dataset.records],
                             [row["image_id"] for row in expected])
            self.assertEqual([record.patient_id for record in loader.dataset.records],
                             [row["patient_id"] for row in expected])
            self.assertEqual(len(next(iter(loader))[1]), min(3, len(expected)))

    def test_augmentation_runs_only_for_train(self) -> None:
        loaders = make_loaders(self.root, self.manifest, device=self.cpu, image_size=4, include_test=True)
        baseline = make_loaders(self.root, self.manifest, device=self.cpu, image_size=4, augment_train=False)
        with patch.object(transforms.RandomHorizontalFlip, "forward", autospec=True,
                          side_effect=lambda _, tensor: tensor.flip(-1)) as flip:
            flipped, _ = loaders.train.dataset[0]
            val, _ = loaders.val.dataset[0]
            test, _ = loaders.test.dataset[0]
        flip.assert_called_once()
        torch.testing.assert_close(flipped, baseline.train.dataset[0][0].flip(-1))
        torch.testing.assert_close(val, baseline.val.dataset[0][0])
        torch.testing.assert_close(test, val)

    def test_default_loaders_do_not_access_test_images(self) -> None:
        for row in self.rows:
            if row["split"] == "test":
                (self.root / row["relative_path"]).unlink()
        loaders = make_loaders(self.root, self.manifest, device=self.cpu)
        self.assertIsNone(loaders.test)
        next(iter(loaders.train))
        next(iter(loaders.val))

    def test_image_cache_decodes_once_and_keeps_augmentation_per_access(self) -> None:
        cached = make_loaders(self.root, self.manifest, device=self.cpu, image_size=4, cache_preprocessed=True)
        plain = make_loaders(self.root, self.manifest, device=self.cpu, image_size=4, augment_train=False)
        expected, label = plain.train.dataset[0]
        flips = iter((False, True))
        with patch.object(Image, "open", wraps=Image.open) as decode, patch.object(
            transforms.RandomHorizontalFlip, "forward",
            side_effect=lambda pixels: pixels.flip(-1) if next(flips) else pixels,
        ) as flip:
            first, first_label = cached.train.dataset[0]
            second, second_label = cached.train.dataset[0]
        decode.assert_called_once()
        self.assertEqual(flip.call_count, 2)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected.flip(-1))
        torch.testing.assert_close(first_label, label)
        torch.testing.assert_close(second_label, label)

    def test_rejects_patient_overlap(self) -> None:
        self.rows[1]["split"] = "test"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "Paciente 001 aparece en train y test"):
            make_loaders(self.root, self.manifest, device=self.cpu)

    def test_rejects_invalid_label(self) -> None:
        self.rows[0]["label"] = "2"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "etiqueta u ojo inválido"):
            make_loaders(self.root, self.manifest, device=self.cpu)

    def test_rejects_parent_path(self) -> None:
        self.rows[0]["relative_path"] = "../outside.png"
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "Ruta relativa inválida"):
            make_loaders(self.root, self.manifest, device=self.cpu)
