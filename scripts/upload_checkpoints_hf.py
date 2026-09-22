"""Upload trained checkpoints (plus their results/config/summary) to a HuggingFace model repository.

    HF_TOKEN=hf_... python scripts/upload_checkpoints_hf.py --root /workspace/outputs_e20 \
        --runs pathq_fast_e20_aux pathq_fast_e20_aux_seed1 pathq_fast_e20_aux_seed2 survpath_e20 \
        --repo sharoonsharif1/PathQFormer-checkpoints

Repository layout mirrors the run directories: <root-name>/<run>/<cohort>/fold_k/best_checkpoint.pt next to
results.json / summary.md / config.yaml. Each best_checkpoint.pt is self-contained (model, pathway tokenizer,
survival bins, gene scaler, gene list), so a fold's model can be rebuilt with `src.training.train.build_model`.
Files already on the Hub with the same content are skipped by the Hub's deduplication.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

MODEL_CARD = """---
license: mit
tags:
  - computational-pathology
  - survival-analysis
  - multimodal
  - tcga
library_name: pytorch
---

# PathQ-Former checkpoints

Trained checkpoints behind the tables in <https://github.com/SharoonSharif/PathQFormer> (REPORT.md).
Protocol: SurvPath 5-fold patient-level splits, disease-specific survival, fixed 20-epoch budget, final checkpoint.

| Folder | Model | Seeds | Cohorts |
|---|---|---|---|
{rows}

Layout: `outputs_e20/<run>/<cohort>/fold_k/best_checkpoint.pt` (DSS) and `outputs_os/<run>/...` (overall-survival endpoint, seed 0) with the run's `results.json`, `summary.md` and
`config.yaml`. A checkpoint holds the model and pathway-tokenizer weights, the survival-bin edges, the gene scaler
and the gene list of its training fold, so it can be reloaded with

```python
import torch, yaml
from src.training.train import build_model, with_defaults
ckpt = torch.load("outputs_e20/pathq_fast_e20_aux/blca/fold_0/best_checkpoint.pt", map_location="cpu", weights_only=False)
```

and the helpers in `scripts/analyze_run.py` (attention export) and `scripts/eval_missing_impute.py`.
Input features are UNI2-h patch embeddings (HF dataset `MahmoodLab/UNI2-h-features`, gated) and SurvPath's RNA
matrices; see the GitHub README for data access. The SurvPath checkpoints were trained with the authors' code
(GPLv3, non-commercial academic use) under the same protocol and are provided for the paired comparison only.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="e.g. /workspace/outputs_e20 (its basename becomes the top folder in the repo)")
    ap.add_argument("--runs", nargs="+", required=True, help="run directories under --root")
    ap.add_argument("--repo", default="sharoonsharif1/PathQFormer-checkpoints")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true", help="create the repo private (default public)")
    ap.add_argument("--no-card", action="store_true", help="do not (re)write README.md")
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("set HF_TOKEN or pass --token")

    root = Path(args.root)
    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    patterns = []
    rows = []
    for run in args.runs:
        rd = root / run
        if not rd.is_dir():
            raise SystemExit(f"{rd} not found")
        patterns += [f"{run}/*/fold_*/best_checkpoint.pt", f"{run}/*/results.json", f"{run}/*/summary.md", f"{run}/*/config.yaml"]
        cohorts = sorted(p.name for p in rd.iterdir() if p.is_dir())
        n_ckpt = len(list(rd.glob("*/fold_*/best_checkpoint.pt")))
        model = "SurvPath (official code)" if run.startswith("survpath") else "PathQ-Former + aux heads" if "aux" in run else "PathQ-Former"
        if "wsi_only" in run:
            model = "PathQ-Former, WSI only"
        if "genomic_only" in run:
            model = "PathQ-Former, RNA only"
        seed = run.rsplit("_seed", 1)[1] if "_seed" in run else "0"
        rows.append(f"| `{root.name}/{run}` | {model} | {seed} | {', '.join(c.upper() for c in cohorts)} ({n_ckpt} checkpoints) |")

    if not args.no_card:
        api.upload_file(path_or_fileobj=MODEL_CARD.format(rows="\n".join(rows)).encode("utf-8"), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model", commit_message="model card")
        print("model card uploaded")

    print(f"uploading {len(args.runs)} runs from {root} -> {args.repo}/{root.name}", flush=True)
    api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(root), path_in_repo=root.name,
                      allow_patterns=patterns, commit_message=f"{root.name}: {' '.join(args.runs)}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
