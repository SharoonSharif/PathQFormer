# PathQ-Former

[![CI](https://github.com/SharoonSharif/PathQFormer/actions/workflows/ci.yml/badge.svg)](https://github.com/SharoonSharif/PathQFormer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22900123.svg)](https://doi.org/10.5281/zenodo.22900123)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Code, configurations, archived per-run results and the experimental record for a **missing-modality audit** of
histology-transcriptomics survival models on five TCGA cohorts, and for PathQ-Former, a query-bottleneck fusion
model of whole-slide-image (WSI) patch embeddings and RNA-seq pathway tokens trained with learned null codes,
modality dropout and auxiliary unimodal heads so that **one checkpoint serves patients with WSI + RNA, WSI only, or
RNA only**.

**Results, protocol and every table:** [REPORT.md](REPORT.md). **Chronological record:** [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). **Revised manuscript source:** [paper/tmlr_v2/](paper/tmlr_v2/) (`main.tex`, figures; the earlier draft is [paper/draft.md](paper/draft.md)). **Checkpoints:** [sharoonsharif1/PathQFormer-checkpoints](https://huggingface.co/sharoonsharif1/PathQFormer-checkpoints) on the HuggingFace Hub (275 fold checkpoints, 18 GB: the final model and every baseline row of the accuracy table below for seed 0, three seeds for PathQ-Former, the single-modality models and the OS endpoint). **Release:** v0.3.0 (public repository since 2026-10-05; see [CITATION.cff](CITATION.cff)); archived versions resolve from the Zenodo concept DOI [10.5281/zenodo.22900123](https://doi.org/10.5281/zenodo.22900123) (latest archived version [v0.2.1](https://github.com/SharoonSharif/PathQFormer/releases/tag/v0.2.1)).

## What the audit found

Every trained checkpoint is re-evaluated on its validation fold with one input removed, with one input permuted
across the validation patients, and with 10-50 % of patients randomly lacking a modality (3 seeds x 5 folds x
5 cohorts, equal 20-epoch budget, final checkpoint, disease-specific survival). Pooled differences below are means
over the 25 seed-averaged (cohort, fold) units with paired t / Wilcoxon p-values
(`results/final/table_missing_modality_seeds.txt`, `table_risk_correlation.txt`).

- **The official SurvPath implementation ranks patients the same way with or without its RNA input.** Permuting the
  RNA across patients changes its C-index by +0.001 (t p 0.36 / W p 0.13); replacing it by the training mean, +0.001.
  Permuting the slide costs it 0.068 (p 0.016 / 0.007) and removing the slide 0.073, leaving 0.47-0.52 on every
  cohort (BLCA 0.486, BRCA 0.520, COADREAD 0.473, HNSC 0.504, STAD 0.493).
- **Null codes and modality dropout remove the collapse; auxiliary heads make the fused output respond to RNA.**
  PathQ-Former w/o aux heads is as blind to RNA removal as SurvPath (+0.000) but loses only 0.030 without the slide
  (SurvPath -0.073; the same checkpoints, one seed, lose 0.069 under mean imputation instead of the null code). With aux heads
  the fused prediction responds to RNA: -0.023 when it is permuted (p 0.075 / 0.063; -0.013 when removed) against
  -0.067 when the slide is permuted. The fused risk stays histology-led (Spearman 0.83 with the checkpoint's own
  RNA-removed risk, 0.45 with its slide-removed risk).
- **One checkpoint covers incomplete patients.** PathQ-Former keeps 0.54-0.62 with either input removed (RNA removed
  0.576-0.600, slide removed 0.538-0.617; three-seed means per cohort) and moves by at most 0.025 when half of the
  validation patients lack a modality (BRCA 0.622 -> 0.597); every change is within one seed standard deviation.
- **Accuracy at equal budget is parity, not superiority** (`table_primary_tests.txt`):

| DSS C-index, 20 epochs, final checkpoint; mean over 3 seeds of the 5-fold mean | BLCA | BRCA | COADREAD | HNSC | STAD | vs SurvPath, 25 seed-averaged folds: delta [95 % bootstrap CI], t p / W p |
|---|---|---|---|---|---|---|
| **PathQ-Former** (aux heads; `pathq_fast_e20_aux`) | **0.630** | 0.622 | 0.613 | 0.582 | 0.572 | +0.036 [-0.004, +0.071], 0.074 / 0.030 (18/25 folds) |
| PathQ-Former w/o aux heads (`pathq_fast_e20`) | 0.609 | 0.611 | **0.638** | 0.575 | **0.593** | +0.037 [+0.009, +0.065], 0.020 / 0.020 (17/25) |
| SurvPath (official code, same protocol) | 0.594 | 0.536 | 0.570 | 0.552 | 0.587 | reference |
| Late fusion (WSI-only + RNA-only PathQ-Former; seeds 0-1) | 0.629 | 0.606 | 0.620 | 0.571 | 0.544 | PathQ-Former vs it: +0.011 [-0.013, +0.034], 0.39 / 0.28 |
| ABMIL (WSI) | 0.566 | 0.575 | 0.587 | 0.562 | 0.553 | +0.001 [-0.024, +0.026], 0.95 / 0.85 |
| SNN (RNA) | 0.590 | 0.558 | 0.572 | 0.535 | 0.549 | -0.007 [-0.060, +0.045], 0.80 / 0.73 |
| RNA MLP | 0.611 | **0.627** | 0.633 | 0.555 | 0.532 | +0.024 [-0.027, +0.077], 0.38 / 0.41 |

Against the RNA MLP the two fusion variants are +0.012 [-0.022, +0.044] (p 0.47 / 0.34) and +0.014 [-0.023, +0.049]
(p 0.47 / 0.37); the MLP scores higher than PathQ-Former on BRCA and COADREAD. No per-cohort DSS difference survives
Holm correction over the five cohorts (BRCA +0.086: raw t p 0.017, Holm 0.084). Earlier versions of this README
reported the same margins over 75 (fold, seed) pairs (t p 0.002-0.007); those pairs are not independent units,
because the three seeds of a fold share its validation patients, so the seed-averaged tests above are the primary
analysis. On overall survival (one seed) the margin over SurvPath is +0.034 [+0.014, +0.054] (p 0.003 / 0.003).

Two further findings from the archived runs: replaying validation-loss early stopping (patience 5) on the stored
per-epoch histories would have selected epoch 1-3 in 93 % of folds and cost PathQ-Former 0.059 C-index, its variant
without aux heads 0.090, SurvPath nothing (+0.000) and ABMIL -0.001 (`table_selection_rules.txt`); and the two models
have the same forward latency (35 vs 36 ms per patient), with a 104 vs 176 ms training step and 16.9 M vs 21.2 M
parameters (`efficiency.md`).

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
modality per sample with p = 0.15 (never both). "PathQ-Former w/o aux heads" is the same model trained with the
auxiliary-head weight set to 0.

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

# PathQ-Former, one cohort, 5-fold CV, seed 0  (~25 min on an RTX PRO 4000 for BLCA; BRCA ~1.5 h)
.venv/Scripts/python.exe -m src.training.train --config configs/protocol_fixed/pathq_fast_e20_aux.yaml --cancer_type blca --device cuda

# any config key can be overridden; numeric strings are coerced
.venv/Scripts/python.exe -m src.training.train --config configs/protocol_fixed/pathq_fast_e20_aux.yaml \
    --cancer_type brca --set seed=1 output_dir=outputs_e20/pathq_fast_e20_aux_seed1 endpoint=os

# a resumable queue (logs in logs/), e.g. all six methods on one cohort
bash scripts/run_experiments.sh configs/protocol_fixed/{pathq_fast_e20,pathq_fast_e20_aux,survpath_e20,abmil_e20,snn_e20,mlp_omics_e20}.yaml
```

Runs resume at two levels: finished folds (`cv_progress.json`) and mid-fold (`latest_checkpoint.pt`).
Rerunning a finished config is a no-op. Every run stores its config, seed, git commit, per-fold metrics,
per-fold predictions and per-epoch validation history.

## 4. Reproduce the paper

### 4.1 Tables from the archived results (about 1 minute, no GPU)

The per-run results of all 204 GPU cohort-level runs and 13 laptop runs (`results.json`, `summary.md`,
`config.yaml`, per-fold metrics, predictions and training histories; no checkpoints) are archived in five tarballs
(about 25 MB): `pod_results/pod_results.tgz`, `pod_results/pull_0918/results_0918a.tgz`,
`pod_results/pull_0918/results_0918b.tgz`, `pod_results/pull_0923/results_0923.tgz` (the GPU campaigns) and
`pod_results/laptop_runs.tgz` (13 BLCA laptop-CPU runs: the validation-loss-era baselines and the
final-configuration ablations). `scripts/reproduce_tables.sh` extracts them on a fresh clone and runs every
analysis script:

```bash
bash scripts/reproduce_tables.sh        # -> results/final/  (also committed, so you can just read them)
```

| Script run by `reproduce_tables.sh` | Output in `results/final/` | Content |
|---|---|---|
| `scripts/seed_table.py` | `table1_dss_20ep_vs_survpath.txt`, `table2_dss_20ep_vs_mlp.txt`, `table_os_20ep.txt`, `table_dss_10ep.txt`, `table_fusion_vs_ensemble.txt`, `table_grid.txt` | C-index per cohort (mean +- sd over seeds of the 5-fold mean) with paired tests over (fold, seed) pairs and, as the primary analysis, over the seed-averaged folds; vs SurvPath, vs the RNA MLP, OS endpoint, 10-epoch budget, late fusion, learning-rate check |
| `scripts/aggregate_results.py` | `summary_{e20,os,ablate,v2}_all_runs.md` | every run with all metrics, missing-modality blocks and per-fold rows |
| `scripts/missing_modality_table.py` | `table_missing_modality_seeds.txt`, `table_final_config_ablations.txt` | the audit over seeds: removed (null code and mean imputation), permuted and randomly missing inputs for PathQ-Former, its variant without aux heads and SurvPath; the same conditions for the five final-configuration ablations on BLCA |
| `scripts/risk_correlation.py` | `table_risk_correlation.txt` | Spearman correlation of PathQ-Former's fused risk with its own single-input risks and with the other models' risks |
| `scripts/posthoc_selection.py` | `table_selection_rules.txt` | replay of validation-loss early stopping, argmin val-loss and the oracle on the stored per-epoch histories (Tables A-D) |
| `scripts/primary_tests.py` | `table_primary_tests.txt`, `table_secondary_metrics.txt` | seed-averaged 25-fold paired tests with 95 % bootstrap CIs and Holm correction over cohorts; Uno's IPCW C-index and integrated Brier score |
| (copied) | `efficiency.md` | parameters, latency, memory vs SurvPath (`scripts/efficiency.py` on a pod) |

The figures of the manuscript are produced from the same archives and the committed analysis files:
`scripts/paper_figures.py --root pod_results/outputs_e20 --out paper/tmlr_v2/figures` (audit deltas with bootstrap
CIs, missing-rate curves over seeds, pooled and per-cohort epoch curves) and
`scripts/paper_figures_km.py --outdir paper/tmlr_v2/figures` (Kaplan-Meier grid from the stored per-fold
predictions, pathway-attention panels from the `analysis/pathway_importance.csv` files that
`scripts/analyze_run.py --attention` writes from the checkpoints). `scripts/make_anonymous_supplement.py --out
supplementary.zip` builds a double-blind supplement from the tracked files (code, configs, tests, scripts,
`results/final/`, the five archives, REPORT and LOG), applies the de-identifying replacements and re-scans the zip
for identifying strings.

### 4.2 Retrain everything

Each cell of the table above is one command; the campaign is `configs/protocol_fixed/{pathq_fast_e20, pathq_fast_e20_aux,
survpath_e20, abmil_e20, snn_e20, mlp_omics_e20}.yaml` x `--cancer_type {blca,brca,coadread,hnsc,stad}` x
`--set seed={0,1,2}` (output_dir suffixed `_seed1`, `_seed2` for seeds 1-2, which is what `scripts/seed_table.py`
expects). The other experiments:

| Experiment | How |
|---|---|
| Audit conditions (removed / permuted inputs, random missingness) | removed and random-missing conditions are produced by every PathQ-Former run (`eval_missing: true`); mean imputation and permutation for any run: `scripts/eval_missing_impute.py <run_dir> --permute` |
| Single-modality PathQ-Former and late fusion | `configs/protocol_fixed/pathq_e20_{wsi,genomic}_only.yaml`, then `scripts/late_fusion.py <run_a> <run_b> --out ...` |
| OS endpoint | same configs with `--set endpoint=os output_dir=outputs_os/<run>` |
| Ablations (bins, pathway sets, patch budget) | `configs/ablation_fixed/pathq_aux_e20_*.yaml` (laptop CPU runs, archived); fusion ablations (K, depth, modality dropout): `configs/ablation_fixed/pathq_fast_e10_*.yaml` |
| Epoch curves (budget choice) | `scripts/epoch_curves.py <roots> --runs pathq_fast_e20 survpath_e20 --epochs 20` |
| Efficiency | `scripts/efficiency.py --embeddings_dir data/embeddings/uni2h/BLCA --out results/efficiency.md` |
| Attention / KM / training-curve figures per run | `scripts/analyze_run.py <run_dir> --attention` (the committed ones in `paper/figures/` came from the Hub checkpoints on a pod) |
| Pooled epoch curves (PNG, earlier draft) | `scripts/epoch_curves.py pod_results/outputs_e20 --runs pathq_fast_e20 pathq_fast_e20_aux survpath_e20 --epochs 20 --plot paper/figures/epoch_curves_pooled.png` |
| Checkpoint upload | `scripts/upload_checkpoints_hf.py --root outputs_e20 --runs ...` |

The GPU campaign took about 210 GPU-hours (RTX PRO 4000 class; training is I/O-bound on slide features)
and about 130 USD on Runpod. `scripts/pod/` holds the exact queue scripts used (Runpod-specific, but
`run_queue_pod.sh` is a plain bash loop over cohorts x seeds x configs that skips finished runs).

### 4.3 SurvPath trained with the recipe (implemented, not run)

`src/models/survpath_recipe.py` (`model_type: survpath_recipe`) wraps the official SurvPath submodules unchanged and
adds PathQ-Former's training recipe: learned null tokens inserted before the co-attention for an absent modality
(32 null patch tokens; one null token per pathway), per-sample modality dropout, and optional auxiliary unimodal
heads. `configs/protocol_fixed/survpath_recipe_e20.yaml` (aux heads, weight 0.5) and
`survpath_recipe_noaux_e20.yaml` (weight 0) mirror `survpath_e20.yaml`; `scripts/pod/batch10.sh` is the queue (three
seeds, five cohorts, then the imputation and permutation evaluations). The model is implemented and unit-tested on
CPU (`tests/test_survpath_recipe.py`), but **its five-cohort training has not been run**: the batch was started on a
pod on 2026-10-05 and stopped after the first fold of BLCA, so no result for it exists in the archives, the tables or
the manuscript.

### 4.4 Protocol (identical for every method)

| Aspect | Choice |
|---|---|
| Unit of analysis | patient (`case_id`); patches of all slides of a patient are concatenated |
| Splits / endpoint | SurvPath's exact 5-fold CV splits; the validation fold is the reported fold; disease-specific survival (DSS); OS as a secondary endpoint |
| Time bins | quartiles of **uncensored training** patients, outer edges +-inf, re-used for validation |
| Gene scaling | MinMax to [-1, 1] fit on the training split |
| Risk score | -sum_t S_t |
| Primary metric | Harrell's C-index on **continuous** time (`sksurv.concordance_index_censored`) |
| Also reported | Uno's IPCW C-index truncated at the 75th percentile of training event times; Brier / IBS and td-AUC on the interior quartile grid; 1000x bootstrap 95 % CI; KM log-rank |
| Model selection | `selection_metric: last` = fixed 20-epoch budget, final checkpoint, all methods. `val_loss` (early stopping) and `val_cindex` exist but were rejected: on cohorts with 5-45 events per fold, val-loss selection picks epoch 1-3 checkpoints (REPORT.md sections 3.2-3.3) |
| Seeds / statistics | `seed + fold`; 3 seeds; one value per (cohort, fold) = mean over seeds, paired t and exact Wilcoxon over the 25 pooled folds with a 95 % percentile-bootstrap CI (10,000 resamples) as the primary test; per-cohort tests (n = 5) Holm-adjusted over cohorts and exploratory; (fold, seed)-pair tests are also printed but are not independent units |
| Audit | the same checkpoint evaluated with each input removed (null code for PathQ-Former, training mean for SurvPath) or permuted across validation patients, with 10-50 % of patients randomly missing one modality (five draws), and the Spearman correlation between its fused and single-input risks |

## 5. Repository layout

```
src/models/       pathqformer.py (query blocks, fusion, null codes, per-sample modality dropout, aux heads),
                  baselines.py (ABMIL, SNN, MLP, official-SurvPath wrapper), survpath_recipe.py (SurvPath + the
                  recipe; implemented, not run), query_block.py, fusion_block.py, survival_head.py (NLL loss, risk),
                  null_tokens.py
src/data/         tcga_dataset.py (patient-level dataset, SurvivalBins, OmicsScaler, RAM cache), pathway_tokenizer.py
src/training/     train.py (CV driver, resume, selection, final evaluation, CLI), evaluate.py (all metrics)
src/utils/        repro.py (seeding, git provenance), visualization.py (KM, pathway importance)
configs/          protocol_fixed/ (the paper, incl. survpath_recipe_*), ablation_fixed/, ablation/, baselines/, legacy/ (May 2026, superseded)
scripts/          run_experiments.sh, reproduce_tables.sh, seed_table.py, aggregate_results.py, missing_modality_table.py,
                  risk_correlation.py, posthoc_selection.py, primary_tests.py, epoch_curves.py, late_fusion.py,
                  eval_missing_impute.py, efficiency.py, recompute_metrics.py, analyze_run.py, paper_figures.py,
                  paper_figures_km.py, make_anonymous_supplement.py, make_dummy_embeddings.py, download_embeddings.py,
                  pod/ (Runpod queues, batch2-batch10)
tests/            46 tests (35 fast; 11 `-m slow` = end-to-end trainer tests on the dummy data, ~10 min CPU)
pod_results/      archived per-run results of the campaign (five tarballs); results/final/ = the regenerated tables
paper/tmlr_v2/    revised manuscript source (main.tex, figures/); paper/draft.md, paper/figures/: the earlier draft
REPORT.md         experimental report; EXPERIMENT_LOG.md: chronological record incl. the protocol audit
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
.venv/Scripts/python.exe -m pytest -m "not slow"     # ~20 s: bins, scaler, loss, model, metrics, dataset, baselines, SurvPath recipe, scripts
.venv/Scripts/python.exe -m pytest                   # + end-to-end trainer tests on the dummy embeddings (~10 min, CPU)
```

The slow suite trains real folds with tiny models and checks config overrides and validation, the weighted
sampler, other endpoints, patients without embeddings, a full CV run and the fold skip on rerun, mid-fold
resume, all three selection rules, seed determinism, the auxiliary loss, every baseline through the trainer,
and the analysis scripts end to end. The SurvPath-recipe tests need `data/survpath_repo` and `einops` and are
skipped otherwise. CI runs the fast suite on every push and the slow suite on request.

## 7. Hardware notes

- Training is I/O-bound: one BLCA epoch is ~25 s on an RTX PRO 4000 with features in RAM, ~150 s streaming
  from network storage. `cache_in_ram: true` with `cache_dtype: float16` keeps a cohort's features in memory,
  capped at `cache_max_fraction` (0.45) of the container's cgroup memory limit, which is what `free` does
  **not** show inside a container.
- A 12-core CPU trains BLCA at ~140 s/epoch (5-fold run ~3 h); the 13 laptop-CPU runs in `pod_results/laptop_runs.tgz`
  were produced that way.
- Peak GPU memory of PathQ-Former on full slides is ~1.2 GiB; any GPU works.

## 8. Citation and licence

Code is released under the [MIT licence](LICENSE); see [CITATION.cff](CITATION.cff). Please cite the Zenodo concept DOI 10.5281/zenodo.22900123 (all versions; a version DOI for v0.3.0 is minted when the release is archived) or 10.5281/zenodo.22900124 (v0.2.1). The SurvPath baseline
imports the authors' code from `data/survpath_repo` at run time, which is GPLv3 and for non-commercial
academic use under the Mahmood Lab's terms; UNI2-h features and TCGA data carry their own access terms.
Nothing from those sources is redistributed here.

## Acknowledgements

Splits, RNA preprocessing, pathway compositions and the SurvPath baseline come from
[mahmoodlab/SurvPath](https://github.com/mahmoodlab/SurvPath) (Jaume et al., CVPR 2024); patch features from
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) (Chen et al., Nature Medicine 2024); the discrete-time
survival loss follows MCAT (Chen et al., ICCV 2021).
