"""Download and extract pre-computed UNI2-h WSI embeddings for a TCGA cohort from HuggingFace.

    python scripts/download_embeddings.py --cancer BRCA                  # -> data/embeddings/uni2h/BRCA/*.h5
    python scripts/download_embeddings.py --cancer STAD --token hf_xxx   # or set HF_TOKEN / run `huggingface-cli login`

The dataset repo (MahmoodLab/UNI2-h-features) is gated: request access on HuggingFace first.
Archives: TCGA/TCGA-<CANCER>.tar.gz (BLCA ~30 GB, BRCA ~65 GB, STAD ~18 GB, COADREAD ~22 GB, HNSC ~17 GB).
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

REPO = "MahmoodLab/UNI2-h-features"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancer", required=True, help="TCGA code, e.g. BLCA, BRCA, STAD, COADREAD, HNSC")
    ap.add_argument("--dest", default="data/embeddings/uni2h")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--keep-archive", action="store_true", help="do not delete the .tar.gz after extraction")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install huggingface_hub") from exc

    cancer = args.cancer.upper()
    dest = Path(args.dest) / cancer
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.h5")):
        print(f"{dest} already has {len(list(dest.glob('*.h5')))} .h5 files; nothing to do")
        return

    print(f"downloading TCGA/TCGA-{cancer}.tar.gz from {REPO} ...", flush=True)
    archive = hf_hub_download(
        repo_id=REPO, filename=f"TCGA/TCGA-{cancer}.tar.gz", repo_type="dataset",
        local_dir=str(Path(args.dest) / "_hf_cache"), token=args.token,
    )
    print(f"extracting {archive} -> {dest}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
        for m in members:
            m.name = Path(m.name).name  # flatten
        tar.extractall(dest, members=members)
    print(f"{len(list(dest.glob('*.h5')))} slides extracted", flush=True)
    if not args.keep_archive:
        os.remove(archive)
        print("archive removed")


if __name__ == "__main__":
    main()
