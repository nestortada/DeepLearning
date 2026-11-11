"""ImageNet initialization and transfer learning for PAPILA."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights, ResNet18_Weights, ResNet34_Weights,
    efficientnet_b0, resnet18, resnet34,
)

from .data import Normalization


RESNET18 = "resnet18"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RESNET34 = "resnet34"
EFFICIENTNET_B0 = "efficientnet_b0"
BACKBONES = {
    RESNET18: (resnet18, WEIGHTS),
    RESNET34: (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    EFFICIENTNET_B0: (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
}
WEIGHT_DIGESTS = {
    RESNET18: WEIGHTS_SHA256,
    RESNET34: "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
    EFFICIENTNET_B0: "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
}


def imagenet_normalization(architecture: str = RESNET18) -> Normalization:
    transform = BACKBONES[architecture][1].transforms()
    return Normalization(tuple(transform.mean), tuple(transform.std))


class PapilaTransferModel(nn.Module):
    """A binary classifier with an optional trainable suffix of the backbone."""

    def __init__(
        self, architecture: str, *, pretrained: bool, tune_blocks: int = 0, freeze_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if tune_blocks not in (0, 1, 2):
            raise ValueError("Se pueden ajustar cero, uno o dos bloques finales.")
        builder, weights = BACKBONES[architecture]
        self.architecture = architecture
        self.backbone = builder(weights=weights if pretrained else None, progress=False)
        self.pretrained_sha256 = None
        if pretrained:
            path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
            with path.open("rb") as stream:
                self.pretrained_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if self.pretrained_sha256 != WEIGHT_DIGESTS[architecture]:
                raise ValueError(f"SHA-256 inesperado para los pesos oficiales: {path}")
        self.freeze_batch_norm = freeze_batch_norm
        self.tune_blocks = tune_blocks
        if architecture == EFFICIENTNET_B0:
            # Keep the same linear head protocol for all three architectures.
            head = nn.Linear(self.backbone.classifier[-1].in_features, 1)
            self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), head) if dropout else head
        else:
            head = nn.Linear(self.backbone.fc.in_features, 1)
            self.backbone.fc = nn.Sequential(nn.Dropout(dropout), head) if dropout else head
        self.configure_tuning(enabled=True)

    def _tuned_modules(self) -> tuple[nn.Module, ...]:
        if not self.tune_blocks:
            return ()
        if self.architecture == EFFICIENTNET_B0:
            return tuple(self.backbone.features)[-(self.tune_blocks + 1):]
        return (self.backbone.layer4,) if self.tune_blocks == 1 else (self.backbone.layer3, self.backbone.layer4)

    def configure_tuning(self, *, enabled: bool) -> None:
        self.tuning_enabled = enabled
        self.backbone.requires_grad_(False)
        self.fc_out.requires_grad_(True)
        if enabled:
            for stage in self._tuned_modules():
                stage.requires_grad_(True)
                if self.freeze_batch_norm:
                    for module in stage.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            module.requires_grad_(False)
        self.train(self.training)

    @property
    def classification_head(self) -> nn.Module:
        if self.architecture == EFFICIENTNET_B0:
            return self.backbone.classifier
        return self.backbone.fc

    @property
    def fc_out(self) -> nn.Linear:
        head = self.classification_head
        return head[-1] if isinstance(head, nn.Sequential) else head

    @property
    def gradcam_layer_name(self) -> str:
        if self.architecture == EFFICIENTNET_B0:
            return "backbone.features[-1]"
        return "backbone.layer4[-1]"

    @property
    def gradcam_layer(self) -> nn.Module:
        if self.architecture == EFFICIENTNET_B0:
            return self.backbone.features[-1]
        return self.backbone.layer4[-1]

    def train(self, mode: bool = True) -> PapilaTransferModel:
        super().train(mode)
        self.backbone.eval()
        self.classification_head.train(mode)
        if mode and self.tuning_enabled and not self.freeze_batch_norm:
            # Update statistics only in the unfrozen suffix. Stochastic depth stays off.
            for stage in self._tuned_modules():
                for module in stage.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.train()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(1)


class PapilaResNet18(PapilaTransferModel):
    """Retain the original class and checkpoint layout for ResNet18 callers."""

    def __init__(
        self, *, pretrained: bool, tune_last_block: bool = False, tune_blocks: int | None = None,
        freeze_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            RESNET18, pretrained=pretrained, tune_blocks=int(tune_last_block) if tune_blocks is None else tune_blocks,
            freeze_batch_norm=freeze_batch_norm, dropout=dropout,
        )
