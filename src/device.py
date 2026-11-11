"""Device and mixed-precision settings for PyTorch training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    amp_enabled: bool
    amp_dtype: torch.dtype = torch.float16
    use_pin_memory: bool = True

    @property
    def pin_memory(self) -> bool:
        return self.device.type == "cuda" and self.use_pin_memory

    def autocast(self) -> torch.autocast:
        """Use the chosen autocast dtype on CUDA when AMP is enabled."""
        return torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)

    def make_grad_scaler(self) -> torch.amp.GradScaler:
        """Create one scaler to reuse throughout a training run."""
        return torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)

    def move_batch(self, batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        return (
            images.to(self.device, non_blocking=self.pin_memory),
            labels.to(self.device, non_blocking=self.pin_memory),
        )


def select_device(
    *, prefer_cuda: bool = True, enable_amp: bool = True, amp_dtype: torch.dtype = torch.float16,
    pin_memory: bool = True,
) -> DeviceConfig:
    """Choose the first visible CUDA device, with CPU as fallback."""
    use_cuda = prefer_cuda and torch.cuda.is_available()
    if use_cuda and enable_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("La GPU no admite AMP BF16.")
    return DeviceConfig(
        torch.device("cuda:0" if use_cuda else "cpu"), use_cuda and enable_amp, amp_dtype, pin_memory,
    )
