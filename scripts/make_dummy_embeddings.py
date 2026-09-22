"""Create random patch "embeddings" for every slide of a cohort so tests, `--smoke` and CI run without TCGA data.

    python scripts/make_dummy_embeddings.py --cancer blca                 # -> data/embeddings/uni2h_dummy/<slide>.pt
    python scripts/make_dummy_embeddings.py --cancer stad --n_patches 64 --dest /tmp/dummy_stad

Each slide listed in SurvPath's metadata CSV gets one float32 tensor of shape (n_patches, dim) saved with
`torch.save`, named after the slide stem exactly as the real UNI2-h files are, so `TCGAMultimodalDataset`
indexes them the same way. Requires the SurvPath repository (splits, metadata) at --survpath_dir:

    git clone --depth 1 https://github.com/mahmoodlab/SurvPath data/survpath_repo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.tcga_dataset import slide_stem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cancer", default="blca", help="TCGA cohort code as in SurvPath: blca, brca, coadread, hnsc, stad")
    ap.add_argument("--survpath_dir", default="data/survpath_repo")
    ap.add_argument("--dest", default="data/embeddings/uni2h_dummy")
    ap.add_argument("--n_patches", type=int, default=230, help="patches per slide (real slides have 1k-40k)")
    ap.add_argument("--dim", type=int, default=1536, help="UNI2-h feature dimension")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    meta_csv = Path(args.survpath_dir) / "datasets_csv" / "metadata" / f"tcga_{args.cancer.lower()}.csv"
    if not meta_csv.exists():
        raise SystemExit(f"{meta_csv} not found; clone SurvPath first (see the module docstring)")
    slides = pd.read_csv(meta_csv)["slide_id"].astype(str).map(slide_stem).drop_duplicates().tolist()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    written = skipped = 0
    for stem in slides:
        out = dest / f"{stem}.pt"
        if out.exists() and not args.force:
            skipped += 1
            continue
        torch.save(torch.randn(args.n_patches, args.dim, generator=gen), out)
        written += 1
    size_mb = sum(p.stat().st_size for p in dest.glob("*.pt")) / 2**20
    print(f"{args.cancer.upper()}: {len(slides)} slides in {meta_csv.name}; wrote {written}, kept {skipped} -> {dest} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
