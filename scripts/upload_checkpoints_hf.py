"""Upload trained checkpoints (plus their results/config/summary) to a HuggingFace model repository.

    HF_TOKEN=hf_... python scripts/upload_checkpoints_hf.py --root /workspace/outputs_e20 \
        --runs pathq_fast_e20_aux pathq_fast_e20_aux_seed1 pathq_fast_e20_aux_seed2 survpath_e20 \
        --repo sharoonsharif1/PathQFormer-checkpoints

Repository layout mirrors the run directories: <root-name>/<run>/<cohort>/fold_k/best_checkpoint.pt next to
results.json / summary.md / config.yaml. Each best_checkpoint.pt is self-contained (model, pathway tokenizer,
survival bins, gene scaler, gene list), so a fold's model can be rebuilt with `src.training.train.build_model`.
Files already on the Hub with the same content are skipped by the Hub's deduplication.

Card only (no checkpoints needed locally; the run table is rebuilt from the files already on the Hub):

    HF_TOKEN=hf_... python scripts/upload_checkpoints_hf.py --card-only            # rewrite README.md on the Hub
    python scripts/upload_checkpoints_hf.py --card-only --dry-run                  # print the card, write nothing
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

# NullQ-Former checkpoints

Trained checkpoints behind the tables in <https://github.com/SharoonSharif/PathQFormer> (REPORT.md).
NullQ-Former was called PathQ-Former until October 2026; the repository name and the `pathq_*` run folders keep the old name.
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


def model_label(run: str) -> str:
    model = "SurvPath (official code)" if run.startswith("survpath") else "NullQ-Former + aux heads" if "aux" in run else "NullQ-Former"
    if "wsi_only" in run:
        model = "NullQ-Former, WSI only"
    if "genomic_only" in run:
        model = "NullQ-Former, RNA only"
    return model


def card_row(top: str, run: str, cohorts: list[str], n_ckpt: int) -> str:
    seed = run.rsplit("_seed", 1)[1] if "_seed" in run else "0"
    return f"| `{top}/{run}` | {model_label(run)} | {seed} | {', '.join(c.upper() for c in cohorts)} ({n_ckpt} checkpoints) |"


def rows_from_hub(api: HfApi, repo: str) -> list[str]:
    """One table row per <top>/<run> folder that holds checkpoints on the Hub, in the Hub's folder order."""
    found: dict[tuple[str, str], dict[str, int]] = {}
    for f in api.list_repo_files(repo, repo_type="model"):
        parts = f.split("/")
        if len(parts) == 5 and parts[4] == "best_checkpoint.pt" and parts[3].startswith("fold_"):
            top, run, cohort = parts[0], parts[1], parts[2]
            found.setdefault((top, run), {}).setdefault(cohort, 0)
            found[(top, run)][cohort] += 1
    return [card_row(top, run, sorted(c), sum(c.values())) for (top, run), c in sorted(found.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="e.g. /workspace/outputs_e20 (its basename becomes the top folder in the repo)")
    ap.add_argument("--runs", nargs="+", default=[], help="run directories under --root")
    ap.add_argument("--repo", default="sharoonsharif1/PathQFormer-checkpoints")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true", help="create the repo private (default public)")
    ap.add_argument("--no-card", action="store_true", help="do not (re)write README.md")
    ap.add_argument("--card-only", action="store_true", help="only rewrite README.md, with the run table rebuilt from the Hub listing")
    ap.add_argument("--dry-run", action="store_true", help="with --card-only: print the card and write nothing")
    args = ap.parse_args()

    if args.card_only:
        api = HfApi(token=args.token)
        card = MODEL_CARD.format(rows="\n".join(rows_from_hub(api, args.repo)))
        if args.dry_run:
            print(card)
            return
        if not args.token:
            raise SystemExit("set HF_TOKEN or pass --token")
        api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model", commit_message="model card: NullQ-Former")
        print("model card uploaded")
        return

    if not args.root or not args.runs:
        raise SystemExit("--root and --runs are required unless --card-only is given")
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
        rows.append(card_row(root.name, run, cohorts, n_ckpt))

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
