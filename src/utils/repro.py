"""Reproducibility helpers: seeding and provenance."""

from __future__ import annotations

import os
import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit(cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def physical_cores() -> int:
    try:
        import psutil  # optional

        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        n = os.cpu_count() or 1
        return max(1, n // 2) if n > 4 else n
