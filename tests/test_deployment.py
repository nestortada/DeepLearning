"""Exercise Render startup and public uploads."""

import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import main


REMOTE = {"RENDER": "true"}


class DeploymentTests(unittest.TestCase):
    def test_render_requires_the_bundled_checkpoint(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {**REMOTE, "GLAUCOAI_MODEL": "resnet34"}):
            spec = replace(main.model_spec(), checkpoint=Path(directory) / "resnet34.pt")
            with patch.object(main, "model_spec", return_value=spec), self.assertLogs(main.logger) as logs:
                with self.assertRaisesRegex(RuntimeError, "demostración requiere"), TestClient(main.create_app()):
                    self.fail("El servicio arrancó sin el checkpoint.")
            self.assertIn("resnet34.pt", " ".join(logs.output))

    def test_public_analyze_accepts_cross_origin_uploads(self):
        process = Mock(return_value=main.Analysis(
            model="resnet34", score_glaucoma=0.55, label=1, classification="glaucoma", threshold=0.48704302310943604,
            heatmap_png="heatmap", overlay_png="overlay", target_layer="layer", has_signal=True,
        ))
        service = SimpleNamespace(model=SimpleNamespace(architecture="resnet34"),
                                  device="cpu", threshold=0.48704302310943604, process=process)
        headers = {"Origin": "https://demo.example", "Content-Type": "image/jpeg"}
        with patch.dict(os.environ, REMOTE), patch.object(main, "load_model", return_value=service):
            with TestClient(main.create_app()) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                preflight = client.options("/api/analyze", headers={
                    "Origin": headers["Origin"], "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(preflight.headers["access-control-allow-origin"], "*")
                response = client.post("/api/analyze", headers=headers, content=b"bytes")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["model"], "resnet34")
                self.assertEqual(response.headers["access-control-allow-origin"], "*")
                self.assertEqual(response.headers["cache-control"], "no-store")
                process.assert_called_once_with(b"bytes", "JPEG", explain=True)
