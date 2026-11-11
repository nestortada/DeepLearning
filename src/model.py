"""The notebook's two-block CNN with a binary logit output."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import torch
from torch import nn
from torch.nn import functional as F

from .pretrained import BACKBONES, RESNET18, PapilaResNet18, PapilaTransferModel


ARCHITECTURE = "papila_cnn"
CHECKPOINT_VERSION = 1
CLASSES = ("sano", "glaucoma")


class PapilaCNN(nn.Module):
    def __init__(self, image_size: int = 96, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * (image_size // 4) ** 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool1(F.relu(self.conv1(images)))
        features = self.pool2(F.relu(self.conv2(features)))
        features = F.relu(self.fc1(torch.flatten(features, 1)))
        return self.fc_out(self.dropout(features)).squeeze(1)


def read_checkpoint(path: Path | BinaryIO) -> dict:
    """Read weights and primitive metadata with CPU mapping for portability."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != CHECKPOINT_VERSION
        or checkpoint.get("architecture", {}).get("name") not in (ARCHITECTURE, *BACKBONES)
        or tuple(checkpoint.get("classes", ())) != CLASSES
    ):
        raise ValueError(f"Checkpoint PAPILA incompatible: {path}")
    return checkpoint


def load_checkpoint(
    path: Path | BinaryIO, device: torch.device | str = "cpu",
) -> tuple[PapilaCNN | PapilaTransferModel, dict]:
    """Restore a model in evaluation mode without downloading initial weights."""
    checkpoint = read_checkpoint(path)
    architecture = checkpoint["architecture"]
    if architecture["name"] == RESNET18:
        model = PapilaResNet18(pretrained=False, dropout=architecture.get("dropout", 0.0))
    elif architecture["name"] in BACKBONES:
        model = PapilaTransferModel(architecture["name"], pretrained=False, dropout=architecture.get("dropout", 0.0))
    else:
        model = PapilaCNN(architecture["image_size"], architecture["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
