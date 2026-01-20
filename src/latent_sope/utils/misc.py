"""Small utilities used across latent_sope.

This repo targets research iteration speed. The helpers below are intentionally
lightweight and dependency-minimal.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Best-effort global seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch might not be installed in some minimal environments
        pass


@dataclass(frozen=True)
class DeviceConfig:
    """Simple device selection helper."""

    device: str = "cuda"
    dtype: str = "float32"

    def torch_dtype(self):
        import torch

        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32


def resolve_device(prefer_cuda: bool = True) -> str:
    """Return "cuda" if available (and preferred), else "cpu"."""
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
