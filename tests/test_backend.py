"""Exercise GlaucoAI's HTTP contract with synthetic images and a frozen ResNet18."""

import base64
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend import main
from src.pretrained import PapilaResNet18


def upload(width=128, height=80, image_format="PNG", *, color=(30, 50, 130), exif=None):
    with Image.new("RGB", (width, height), color) as image, BytesIO() as stream:
        ImageDraw.Draw(image).rectangle((width // 2, 0, width - 1, height // 2), fill=(230, 190, 100))
        image.save(stream, format=image_format, **({"exif": exif} if exif else {}))
        return stream.getvalue()


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            model = PapilaResNet18(pretrained=False).eval().requires_grad_(False)
        model.fc_out.weight.fill_(1 / 512)
        model.fc_out.bias.fill_(-3)
        frozen = main.ROOT / "configs" / "papila-internal-test.json"
        metadata = json.loads(frozen.read_text(encoding="utf-8"))["models"]["ResNet18"]["metadata"]
        cls.service = main.ModelService(model, metadata, 0.35, torch.device("cpu"))

    def setUp(self):
        loader = self.enterContext(patch.object(main, "load_model", return_value=self.service))
        self.client = self.enterContext(TestClient(main.app))
        loader.assert_called_once_with()

    def post(self, path, content, media_type="image/png"):
        return self.client.post(path, content=content, headers={"Content-Type": media_type})

    def assert_pngs(self, result, size):
        for key in ("heatmap_png", "overlay_png"):
            data = base64.b64decode(result[key], validate=True)
            with Image.open(BytesIO(data)) as image:
                self.assertEqual((image.format, image.size), ("PNG", size))
                image.verify()

    def test_health_and_one_forward_per_analysis(self):
        health = self.client.get("/health").json()
        self.assertEqual(health, {"status": "ok", "model_ready": True, "model": "resnet18",
                                  "device": "cpu", "threshold": 0.35})
        with patch.object(self.service.model, "forward", wraps=self.service.model.forward) as forward:
            analysis = self.post("/analyze", upload())
            self.assertEqual(analysis.status_code, 200, analysis.text[:160])
            self.assertEqual(analysis.headers["cache-control"], "no-store")
            forward.assert_called_once()
        data = analysis.json()
        prediction = self.post("/predict", upload()).json()
        self.assertEqual(set(prediction), {"model", "label", "classification", "threshold", "score_glaucoma"})
        self.assertEqual(prediction["model"], "resnet18")
        self.assertEqual(data["model"], prediction["model"])
        self.assertAlmostEqual(prediction["score_glaucoma"], data["score_glaucoma"], places=6)
        self.assertEqual(prediction["label"], int(data["score_glaucoma"] >= 0.35))
        self.assertEqual(data["classification"], ("sano", "glaucoma")[prediction["label"]])
        self.assertEqual(data["target_layer"], "backbone.layer4[-1]")
        self.assertTrue(data["has_signal"])
        self.assert_pngs(data, (128, 80))

    def test_explain_applies_orientation_and_caps_preview_size(self):
        exif = Image.Exif()
        exif[274] = 6
        response = self.post("/explain", upload(1280, 640, "JPEG", exif=exif), "image/jpeg")
        self.assertEqual(response.status_code, 200, response.text[:160])
        data = response.json()
        self.assertEqual(set(data), {"target_layer", "has_signal", "heatmap_png", "overlay_png"})
        self.assert_pngs(data, (512, 1024))

    def test_invalid_mime_bytes_empty_body_and_dimensions(self):
        for payload, media_type, status in (
            (b"private filename", "text/plain", 415), (b"", "image/png", 422),
            (b"private filename", "image/jpeg", 422), (upload(), "image/jpeg", 422),
            (upload(4097, 16), "image/png", 413), (upload(3000, 2100), "image/png", 413),
            (upload(512, 1), "image/png", 422),
        ):
            with self.subTest(media_type=media_type, size=len(payload), status=status):
                response = self.post("/analyze", payload, media_type)
                self.assertEqual(response.status_code, status)
                self.assertNotIn("private filename", response.text)
                self.assertEqual(response.headers["cache-control"], "no-store")

    def test_content_limit_with_and_without_content_length(self):
        with patch.object(main, "MAX_UPLOAD_BYTES", 20):
            self.assertEqual(self.post("/analyze", b"x" * 21).status_code, 413)
            response = self.post("/analyze", iter([b"x" * 11, b"x" * 10]))
        self.assertEqual(response.request.headers.get("transfer-encoding"), "chunked")
        self.assertEqual(response.status_code, 413)

    def test_concurrent_analyses_keep_maps_isolated_and_model_unchanged(self):
        images = (upload(128, 80), upload(81, 131, color=(180, 100, 40)))
        state = {name: tensor.clone() for name, tensor in self.service.model.state_dict().items()}
        layer = self.service.model.backbone.layer4[-1]
        hooks = tuple(layer._forward_hooks)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.post, "/analyze", image) for image in images]
            for size, image, future in zip(((128, 80), (81, 131)), images, futures):
                response = future.result(timeout=30)
                self.assertEqual(response.status_code, 200, response.text[:160])
                self.assert_pngs(response.json(), size)
                self.assertEqual(response.json(), self.post("/analyze", image).json())
        self.assertEqual(tuple(layer._forward_hooks), hooks)
        for name, tensor in self.service.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)

    def test_internal_error_omits_request_and_exception_details(self):
        with patch.object(self.service, "process", side_effect=RuntimeError("private filename")):
            with self.assertLogs(main.logger, level="ERROR") as logs:
                response = self.post("/analyze", upload())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private filename", response.text + " ".join(logs.output))


class ModelStartupTests(unittest.TestCase):
    def test_bad_checkpoint_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            path.write_bytes(b"corrupted checkpoint")
            spec = main.ModelSpec("resnet34", path, "0" * 64, 0.5)
            with patch.object(main, "model_spec", return_value=spec), patch.object(main, "load_checkpoint") as deserialize:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    main.load_model()
            deserialize.assert_not_called()

    def test_model_selection_rejects_unknown_architectures(self):
        with patch.dict(os.environ, {"GLAUCOAI_MODEL": "unknown"}), patch.object(main, "load_service") as loader:
            with self.assertRaisesRegex(ValueError, "GLAUCOAI_MODEL"):
                main.load_model()
            loader.assert_not_called()

    def test_checkpoint_architecture_must_match_the_registry(self):
        service = SimpleNamespace(model=SimpleNamespace(architecture="resnet18"))
        with patch.dict(os.environ, {"GLAUCOAI_MODEL": "resnet34"}), patch.object(main, "load_service", return_value=service):
            with self.assertRaisesRegex(ValueError, "arquitectura configurada"):
                main.load_model()

    def test_health_and_analysis_return_503_when_checkpoint_is_missing(self):
        with patch.object(main, "load_model", side_effect=FileNotFoundError):
            with self.assertLogs(main.logger, level="ERROR"), TestClient(main.app) as client:
                response = client.get("/health")
                self.assertEqual(response.status_code, 503)
                self.assertFalse(response.json()["model_ready"])
                response = client.post("/analyze", content=upload(), headers={"Content-Type": "image/png"})
                self.assertEqual(response.status_code, 503)
