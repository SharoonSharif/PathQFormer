# PathQ-Former

[![CI](https://github.com/SharoonSharif/PathQFormer/actions/workflows/ci.yml/badge.svg)](https://github.com/SharoonSharif/PathQFormer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Query-based multimodal fusion of whole-slide-image (WSI) patch embeddings and RNA-seq pathway tokens for
cancer survival prediction, with **missing-modality robustness** built in. One checkpoint serves patients
with WSI + RNA, WSI only, or RNA only.

**Results, protocol and every table:** [REPORT.md](REPORT.md). **Chronological record:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

| DSS C-index, 20 epochs, final checkpoint, 3 seeds x 5 folds | BLCA | BRCA | COADREAD | HNSC | STAD | pooled vs SurvPath (75 pairs) |
|---|---|---|---|---|---|---|
| **PathQ-Former + aux heads** (final config) | **0.630** | 0.622 | 0.613 | **0.582** | 0.572 | **+0.036**, t p 0.007, Wilcoxon p 0.002 |
| PathQ-Former | 0.609 | 0.611 | **0.638** | 0.575 | **0.593** | **+0.037**, t p 0.002, Wilcoxon p 0.005 |
| SurvPath (official code, same protocol) | 0.594 | 0.536 | 0.570 | 0.552 | 0.587 | reference |
| ABMIL (WSI) / SNN (RNA) / MLP (RNA) | 0.566 / 0.590 / 0.611 | 0.575 / 0.558 / **0.627** | 0.587 / 0.572 / 0.633 | 0.562 / 0.535 / 0.555 | 0.553 / 0.549 / 0.532 | +0.001 / -0.007 / +0.024, all n.s. |

With either modality removed at test time the final model keeps 0.54-0.67; SurvPath is unchanged without RNA
(it ignores it) and drops to chance without WSI. Same forward latency as SurvPath, 1.7x faster training
step, 20 % fewer parameters. Honest caveat: PathQ-Former is within noise of a tuned RNA-only MLP pooled
(+0.012); the claim is "at least as good as the best unimodal model on every cohort, with one robust network".

```
WSI patches (frozen UNI2-h, N x 1536)      RNA-seq (4,999 genes) -> 275 pathway tokens
            |                                              |
   Histology query block (K=32)                Genomic query block (K=32)
   cross-attn -> self-attn -> FFN              cross-attn -> self-attn -> FFN
            |  Z_h (32 x 256)                              |  Z_g (32 x 256)
            +---------------------- [Z_h ; Z_g] -----------+
                                       |
                       Cross-modal fusion Transformer (2 layers)
                                       |
                        attention pooling -> 4 discrete hazards -> NLL-survival loss
         (+ auxiliary unimodal survival heads on Z_h and Z_g, weight 0.5)
```

An absent modality (train **or** test) is replaced by a learned null code, and training drops each present
modality per sample with p = 0.15 (never both).

---

## 1. Install

Python 3.10-3.12 (scikit-survival has no wheels for newer versions yet). Pinned versions that produced the
reported numbers are in `requirements-lock.txt`; `requirements.txt` gives the loose ranges.

```bash
git clone https://github.com/SharoonSharif/PathQFormer && cd PathQFormer
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel, e.g. .../whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt einops                                 # einops: needed by the SurvPath baseline
git clone --depth 1 https://github.com/mahmoodlab/SurvPath data/survpath_repo                      # splits, metadata, RNA, pathway compositions (250 MB)
python scripts/make_dummy_embeddings.py --cancer blca                                              # random features: tests and --smoke run without TCGA data
python -m pytest -m "not slow"                                                                     # ~20 s
```

Windows: use `.venv/Scripts/python.exe` in place of `.venv/bin/python` (all commands below are written
that way because the reference runs were launched from Windows). `make venv install survpath dummy-data test`
does the same on any platform with GNU make. A CUDA `Dockerfile` is included (`docker build -t pathqformer .`).

## 2. Data

| What | Where | Size | Access |
|---|---|---|---|
| Splits, clinical metadata, RNA-seq (4,999 genes), pathway compositions | `data/survpath_repo/` = the [SurvPath](https://github.com/mahmoodlab/SurvPath) repository | 250 MB | public, GPLv3 (their terms) |
| UNI2-h patch features per slide (`.h5`: `features` (1, N, 1536), `coords` (1, N, 2)) | `data/embeddings/uni2h/` from HF dataset [MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features), archives `TCGA/TCGA-<CANCER>.tar.gz` | 156 GB for the five cohorts (BRCA largest) | gated: request access on HuggingFace, then |

```bash
export HF_TOKEN=hf_...                                        # or: cp .env.example .env and fill it in
python scripts/download_embeddings.py --cancer BLCA           # BLCA, BRCA, STAD, COADREAD (COAD+READ), HNSC
```

Any directory layout under `embeddings_dir` works (files are indexed recursively by slide stem; `.h5` and
`.pt` are both accepted). Patients whose slides are missing are dropped with a warning and counted in
`results.json`. TCGA data are subject to the [NIH GDC data use policies](https://gdc.cancer.gov/access-data/data-access-policies).

## 3. Run

```bash
# smoke test of the whole pipeline on the dummy data (1 epoch, ~90 s CPU)
.venv/Scripts/python.exe -m src.training.train --config configs/blca_hybrid_v2.yaml --smoke

# the final configuration, one cohort, 5-fold CV, seed 0  (~25 min on an RTX PRO 4000 for BLCA; BRCA ~1.5 h)
.venv/Scripts/python.exe -m src.training.train --config configs/protocol_fixed/pathq_fast_e20_aux.yaml --cancer_type blca --device cuda

# any config key can be overridden; numeric strings are coerced
.venv/Scripts/python.exe -m src.training.train --config configs/protocol_fixed/pathq_fast_e20_aux.yaml \
    --cancer_type brca --set seed=1 output_dir=outputs_e20/pathq_fast_e20_aux_seed1 endpoint=os

# a resumable queue (logs in logs/), e.g. all six methods on one cohort
bash scripts/run_experiments.sh configs/protocol_fixed/{pathq_fast_e20,pathq_fast_e20_aux,survpath_e20,abmil_e20,snn_e20,mlp_omics_e20}.yaml
```

Runs resume at two levels: finished folds (`cv_progress.json`) and mid-fold (`latest_checkpoint.pt`).
Rerunning a finished config is a no-op. Every run stores its config, seed, git commit and per-fold metrics.

## 4. Reproduce the paper

### 4.1 Tables from the archived results (1 minute, no GPU)

The per-run results of all 180 GPU runs (`results.json`, `summary.md`, `config.yaml`, per-fold metrics and
predictions; no checkpoints) are archived in `pod_results/*.tgz` (18 MB).

```bash
bash scripts/reproduce_tables.sh        # -> results/final/  (also committed, so you can just read them)
```

| File in `results/final/` | Report section |
|---|---|
| `table1_dss_20ep_vs_survpath.txt` | Table 1: six methods, five cohorts, 3 seeds, paired tests vs SurvPath |
| `table2_dss_20ep_vs_mlp.txt` | Table 2: same runs, RNA MLP as reference |
| `table_os_20ep.txt` | overall-survival endpoint |
| `table_dss_10ep.txt` | 10-epoch budget (protocol robustness) |
| `summary_*_all_runs.md` | every run with all metrics, missing-modality columns and per-fold rows |
| `efficiency.md` | parameters, latency, memory vs SurvPath |

### 4.2 Retrain everything

Each cell of Table 1 is one command; the campaign is `configs/protocol_fixed/{pathq_fast_e20, pathq_fast_e20_aux,
survpath_e20, abmil_e20, snn_e20, mlp_omics_e20}.yaml` x `--cancer_type {blca,brca,coadread,hnsc,stad}` x
`--set seed={0,1,2}` (output_dir suffixed `_seed1`, `_seed2` for seeds 1-2, which is what `scripts/seed_table.py`
expects). The other experiments:

| Experiment | How |
|---|---|
| Missing-modality Table 3 | produced by every PathQ-Former run (`eval_missing: true`); baselines: `scripts/eval_missing_impute.py <run_dir>` |
| Single-modality PathQ-Former and late fusion | `configs/protocol_fixed/pathq_e20_{wsi,genomic}_only.yaml`, then `scripts/late_fusion.py <run_a> <run_b> --out ...` |
| OS endpoint | same configs with `--set endpoint=os output_dir=outputs_os/<run>` |
| Ablations (bins, pathway sets, patch budget) | `configs/ablation_fixed/pathq_aux_e20_*.yaml`; fusion ablations (K, depth, modality dropout): `configs/ablation_fixed/pathq_fast_e10_*.yaml` |
| Epoch curves (budget choice) | `scripts/epoch_curves.py <roots> --runs pathq_fast_e20 survpath_e20 --epochs 20` |
| Efficiency | `scripts/efficiency.py --embeddings_dir data/embeddings/uni2h/BLCA --out results/efficiency.md` |
| Attention / KM / training-curve figures | `scripts/analyze_run.py <run_dir> --attention` |

The full campaign took about 210 GPU-hours (RTX PRO 4000 class; training is I/O-bound on slide features)
and about 130 USD on Runpod. `scripts/pod/` holds the exact queue scripts used (Runpod-specific, but
`run_queue_pod.sh` is a plain bash loop over cohorts x seeds x configs that skips finished runs).

### 4.3 Protocol (identical for every method)

| Aspect | Choice |
|---|---|
| Unit of analysis | patient (`case_id`); patches of all slides of a patient are concatenated |
| Splits / endpoint | SurvPath's exact 5-fold CV splits; disease-specific survival (DSS); OS as a secondary endpoint |
| Time bins | quartiles of **uncensored training** patients, outer edges +-inf, re-used for validation |
| Gene scaling | MinMax to [-1, 1] fit on the training split |
| Risk score | -sum_t S_t |
| Primary metric | Harrell's C-index on **continuous** time (`sksurv.concordance_index_censored`) |
| Also reported | Uno's IPCW C-index truncated at the 75th percentile of training event times; Brier / IBS and td-AUC on the interior quartile grid; 1000x bootstrap 95 % CI; KM log-rank |
| Model selection | `selection_metric: last` = fixed 20-epoch budget, final checkpoint. `val_loss` (early stopping) and `val_cindex` exist but were rejected: on cohorts with 7-16 events per fold, val-loss selection picked epoch 1-3 checkpoints for every method (REPORT.md section 3.2) |
| Seeds / statistics | `seed + fold`; 3 seeds; paired t and Wilcoxon over (fold, seed) pairs |
| Missing modality | same checkpoint evaluated WSI-only, RNA-only and with 10-50 % of patients randomly missing one modality |

## 5. Repository layout

```
src/models/       pathqformer.py (query blocks, fusion, null codes, per-sample modality dropout, aux heads),
                  baselines.py (ABMIL, SNN, MLP, official-SurvPath wrapper), query_block.py, fusion_block.py,
                  survival_head.py (NLL loss, risk), null_tokens.py
src/data/         tcga_dataset.py (patient-level dataset, SurvivalBins, OmicsScaler, RAM cache), pathway_tokenizer.py
src/training/     train.py (CV driver, resume, selection, final evaluation, CLI), evaluate.py (all metrics)
src/utils/        repro.py (seeding, git provenance), visualization.py (KM, pathway importance)
configs/          protocol_fixed/ (the paper), ablation_fixed/, ablation/, baselines/, legacy/ (May 2026, superseded)
scripts/          run_experiments.sh, seed_table.py, aggregate_results.py, epoch_curves.py, late_fusion.py,
                  eval_missing_impute.py, efficiency.py, recompute_metrics.py, analyze_run.py,
                  make_dummy_embeddings.py, download_embeddings.py, reproduce_tables.sh, pod/ (Runpod queues)
tests/            34 tests; `-m slow` = end-to-end trainer tests on the dummy data (~10 min CPU)
pod_results/      archived per-run results of the campaign (tarballs); results/final/ = the regenerated tables
REPORT.md         final experimental report; EXPERIMENT_LOG.md: chronological record incl. the protocol audit
```

Output of a run:

```
<output_dir>/<cancer>/
├── config.yaml                 resolved config
├── cv_progress.json            fold -> C-index (+ details); drives resume
├── results.json                config, git commit, seeds, per-fold metrics, aggregate mean/std/95% CI
├── summary.md                  human-readable tables (all metrics, missing-modality blocks, per-fold rows)
└── fold_k/
    ├── best_checkpoint.pt      model + tokenizer + bins + scaler + gene list (self-contained for inference)
    ├── fold_results.json       metrics per test-time condition, partial-missing curve, per-epoch history
    └── predictions_{both,wsi_only,genomic_only}.csv   per-patient risk and S_t
```

## 6. Tests

```bash
.venv/Scripts/python.exe -m pytest -m "not slow"     # ~20 s: bins, scaler, loss, model, metrics, dataset, baselines, scripts
.venv/Scripts/python.exe -m pytest                   # + end-to-end trainer tests on the dummy embeddings (~10 min, CPU)
```

The slow suite trains real folds with tiny models and checks config overrides and validation, the weighted
sampler, other endpoints, patients without embeddings, a full CV run and the fold skip on rerun, mid-fold
resume, all three selection rules, seed determinism, the auxiliary loss, every baseline through the trainer,
and the analysis scripts end to end. CI runs the fast suite on every push and the slow suite on request.

## 7. Hardware notes

- Training is I/O-bound: one BLCA epoch is ~25 s on an RTX PRO 4000 with features in RAM, ~150 s streaming
  from network storage. `cache_in_ram: true` with `cache_dtype: float16` keeps a cohort's features in memory,
  capped at `cache_max_fraction` (0.45) of the container's cgroup memory limit, which is what `free` does
  **not** show inside a container.
- A 12-core CPU trains BLCA at ~140 s/epoch (5-fold run ~3 h); the laptop-CPU ablations in the report were
  produced that way.
- Peak GPU memory of the final model on full slides is ~1.2 GiB; any GPU works.

## 8. Citation and licence

Code is released under the [MIT licence](LICENSE); see [CITATION.cff](CITATION.cff). The SurvPath baseline
imports the authors' code from `data/survpath_repo` at run time, which is GPLv3 and for non-commercial
academic use under the Mahmood Lab's terms; UNI2-h features and TCGA data carry their own access terms.
Nothing from those sources is redistributed here.

## Acknowledgements

Splits, RNA preprocessing, pathway compositions and the SurvPath baseline come from
[mahmoodlab/SurvPath](https://github.com/mahmoodlab/SurvPath) (Jaume et al., CVPR 2024); patch features from
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (Chen et al., Nature Medicine 2024); the discrete-time
survival loss follows MCAT (Chen et al., ICCV 2021).
