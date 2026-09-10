# PathQ-Former

Query-based multimodal fusion of whole-slide-image (WSI) patch embeddings and genomic
pathway tokens for cancer survival prediction, with **missing-modality robustness** built in.

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
```

An absent modality (train **or** test) is replaced by a learned null code, and training drops
each present modality per sample with p = 0.15 (never both), so one model serves patients with
WSI + RNA, WSI only, or RNA only without retraining.

## Evaluation protocol (v2, 2026-09-10)

Aligned with MCAT / SurvPath so numbers are comparable to the literature:

| Aspect | Choice |
|---|---|
| Unit of analysis | patient (`case_id`); patches of all slides of a patient are concatenated |
| Splits | SurvPath's exact 5-fold CV splits, disease-specific survival (DSS) by default |
| Time bins | quartiles of **uncensored training** patients, outer edges ±inf, re-used for validation |
| Gene scaling | MinMax to [-1, 1] fit on the training split (SurvPath) |
| Risk score | −Σ<sub>t</sub> S<sub>t</sub> (MCAT/SurvPath) |
| C-index | Harrell's on **continuous** time (`sksurv.concordance_index_censored`) + IPCW variant |
| Also reported | Brier / IBS, mean time-dependent AUC, 1000× bootstrap 95% CI, KM log-rank (median split) |
| Model selection | validation **loss** (selecting on val C-index and reporting it is optimistic; both are logged) |
| Seeds | `seed + fold`, stored with results together with the git commit |
| Missing modality | same checkpoint evaluated WSI-only, genomics-only and with 10/20/30/50 % of patients randomly missing one modality |

The May-2026 numbers in `EXPERIMENT_LOG.md` were produced by a pipeline that computed the
C-index on the 4 discrete bins, fit bins separately on train and validation, used slide-level
samples and unscaled genes; they are **not comparable** to v2 or to published baselines.

## Quick start

```bash
# environment (Python 3.12; scikit-survival has no wheels for 3.14 yet)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# tests (≈20 s, CPU) and an end-to-end smoke run on dummy embeddings (≈90 s)
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m src.training.train --config configs/blca_hybrid_v2.yaml --smoke

# the real thing: 5-fold CV on BLCA (≈2 h on a 12-core CPU, ≈1 h on a T4)
.venv/Scripts/python.exe -m src.training.train --config configs/blca_hybrid_v2.yaml

# override any config key from the command line
.venv/Scripts/python.exe -m src.training.train --config configs/blca_hybrid_v2.yaml --set num_queries=64 output_dir=outputs_v2/k64

# a queue of runs (resumable; logs in logs/), then a cross-run table with paired tests
bash scripts/run_experiments.sh configs/blca_hybrid_v2.yaml configs/ablation/blca_wsi_only.yaml
.venv/Scripts/python.exe scripts/aggregate_results.py outputs_v2 --compare hybrid
```

Runs resume at two levels: finished folds (`cv_progress.json`) and mid-fold (`latest_checkpoint.pt`).

## Data setup

```
data/
├── survpath_repo/        git clone https://github.com/mahmoodlab/SurvPath   (splits, metadata, RNA, pathway compositions)
└── embeddings/uni2h/     TCGA-<CANCER>.tar.gz from HF dataset MahmoodLab/UNI2-h-features (gated), extracted: one .h5 per slide
```

Each `.h5` holds `features` (1, N, 1536) and `coords` (1, N, 2). Any directory layout under
`embeddings_dir` works (files are indexed recursively by slide stem). `data/embeddings/uni2h_dummy/`
contains random 230×1536 tensors for every BLCA slide so tests and `--smoke` run without the real data.

## Outputs

```
<output_dir>/<cancer>/
├── config.yaml                 resolved config
├── cv_progress.json            fold -> C-index (+ details); drives resume
├── results.json                everything: config, git commit, per-fold metrics, aggregate mean/std/95% CI
├── summary.md                  human-readable tables
└── fold_k/
    ├── best_checkpoint.pt      model + tokenizer + bins + scaler + gene list (self-contained for inference)
    ├── fold_results.json       metrics per test-time condition, partial-missing curve, per-epoch history
    └── predictions_{both,wsi_only,genomic_only}.csv   per-patient risk and S_t
```

## Configs

| File | Purpose |
|---|---|
| `configs/blca_hybrid_v2.yaml` | main run: lr 5e-5, dropout 0.25, grad-accum 8, warmup 1, patience 5 (min 6 epochs) |
| `configs/blca_baseline_v2.yaml` | Phase-1 hyper-parameters under the v2 protocol (paired comparison) |
| `configs/ablation/blca_wsi_only.yaml`, `blca_genomic_only.yaml` | component ablation: one modality at train and test |
| `configs/ablation/blca_k16.yaml`, `blca_k64.yaml` | query-count ablation |
| `configs/baselines/blca_survpath.yaml` | **official SurvPath** model (imported from `data/survpath_repo`) on the same 275 pathways |
| `configs/baselines/blca_abmil.yaml`, `blca_snn.yaml`, `blca_mlp_omics.yaml` | WSI-only gated-attention MIL, genomics-only SNN / MLP |
| `configs/legacy/` | the May-2026 configs, kept for the record |

Baselines run through the *same* trainer (`model_type: survpath | abmil | snn | mlp_omics`) on the same
patients, splits, bins, gene scaling, selection rule and metrics; their hyper-parameters follow the
SurvPath repository run scripts (RAdam, lr 5e-4 / 1e-3, wd 1e-4, `nll_alpha: 0.5`, class-balanced
`weighted_sample: true`, 4096 training patches). `scripts/aggregate_results.py --compare hybrid` adds
paired t / Wilcoxon tests against PathQ-Former over the five folds.

Other cancers: `--cancer_type brca|stad|coadread|hnsc` after downloading their embeddings.
Other endpoints: `--set endpoint=os` or `pfi`.

## Layout

```
src/models/      pathqformer.py (fusion + null codes + per-sample modality dropout), query_block.py,
                 fusion_block.py, survival_head.py (NLL loss, risk), null_tokens.py
src/data/        tcga_dataset.py (patient-level dataset, SurvivalBins, OmicsScaler), pathway_tokenizer.py
src/training/    train.py (CV driver, resume, selection, final evaluation), evaluate.py (all metrics)
src/utils/       repro.py (seeding, git provenance), visualization.py (KM, pathway importance)
scripts/         run_experiments.sh, aggregate_results.py, package_src.py, download_data.py
tests/           pytest suite (bins, scaler, loss, model, metrics, dataset)
```
