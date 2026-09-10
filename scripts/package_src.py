"""Zip the source tree (no data, outputs, venv, git) for upload to Colab / a GPU pod.

    python scripts/package_src.py            -> PathQFormer_src_<gitsha>.zip in the repo root
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "scripts", "tests", "notebooks", "requirements.txt", "pyproject.toml", "README.md", "PLAN.md", "EXPERIMENT_LOG.md"]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}


def main() -> None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "nogit"
    except Exception:  # noqa: BLE001
        sha = "nogit"
    out = ROOT / f"PathQFormer_src_{sha}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            path = ROOT / item
            if path.is_file():
                zf.write(path, f"PathQFormer/{item}")
            elif path.is_dir():
                for f in path.rglob("*"):
                    if f.is_file() and not (EXCLUDE_DIRS & set(f.parts)) and f.suffix not in (".pyc", ".pt"):
                        zf.write(f, f"PathQFormer/{f.relative_to(ROOT)}")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
