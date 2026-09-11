# PathQ-Former Experiment Log

## Project Summary
**PathQ-Former**: Q-Former-based multimodal fusion for WSI + genomics cancer survival prediction.
- **Architecture**: Modality-specific query blocks (K=32 learnable queries) → cross-modal fusion Transformer → NLL-Survival head
- **WSI Encoder**: UNI2-h (ViT-H/14, 1536-d, frozen pre-extracted embeddings)
- **Genomic Input**: 4,999 genes → 275 pathways (Reactome + MSigDB Hallmarks via SurvPath compositions)
- **Total trainable params**: ~16.7M (10.4M pathway tokenizer + 6.3M PathQ-Former)
- **Evaluation**: 5-fold stratified CV using SurvPath's exact splits, C-Index metric

---

## Experiment 1: BLCA Baseline
**Date**: 2026-05-19
**Cancer type**: BLCA (Bladder Urothelial Carcinoma)
**Slides**: 423 | **Train/Val**: ~346/77 per fold

### Config
| Parameter | Value |
|-----------|-------|
| LR | 2e-4 |
| Weight decay | 1e-5 |
| Dropout | 0.1 |
| Modality dropout | 0.15 |
| Batch size | 1 |
| Grad accumulation | 1 (none) |
| Warmup | None |
| Patience | 5 |
| Max epochs | 20 |
| Num queries (K) | 32 |
| Hidden dim | 256 |
| Query layers | 2 |
| Fusion layers | 2 |

### Results
| Fold | C-Index | Peak Epoch | Stopped At |
|------|---------|------------|------------|
| 0 | 0.5756 | 3 | 8 |
| 1 | 0.6340 | 2 | 7 |
| 2 | 0.5931 | 2 | 7 |
| 3 | 0.6062 | 8 | 13 |
| 4 | 0.6173 | 1 | 6 |
| **Mean** | **0.6052 ± 0.0200** | | |

### Observations
- Consistent overfitting pattern: C-Index peaks at epoch 2-3 then declines rapidly
- Train loss drops while val loss rises — classic overfitting
- LR too high (2e-4) with batch_size=1 causes noisy, overshooting updates
- Baseline is competitive with MCAT (~0.58-0.61) but below SurvPath/MMP

---

## Experiment 2: BLCA Tuned
**Date**: 2026-05-19
**Cancer type**: BLCA

### Config Changes (vs Baseline)
| Parameter | Baseline | Tuned | Rationale |
|-----------|----------|-------|-----------|
| LR | 2e-4 | **5e-5** | Reduce overshooting |
| Weight decay | 1e-5 | **1e-4** | Stronger L2 regularization |
| Dropout | 0.1 | **0.25** | Reduce overfitting |
| Grad accumulation | 1 | **8** | Effective batch_size=8, smoother gradients |
| Warmup | None | **2 epochs** | Linear LR warmup prevents early instability |
| Patience | 5 | **3** | Stop sooner since improvements are gradual |

### Results
| Fold | C-Index | Peak Epoch | Stopped At | vs Baseline |
|------|---------|------------|------------|-------------|
| 0 | 0.7068 | 8 | 11 | +0.1312 |
| 1 | 0.6606 | 4 | 7 | +0.0266 |
| 2 | 0.6083 | 1 | 4 | +0.0152 |
| 3 | 0.3977 | 2 | 5 | -0.2085 |
| 4 | 0.7025 | 5 | 8 | +0.0852 |
| **Mean** | **0.6152 ± 0.1144** | | | **+0.0100** |

### Observations
- 4 of 5 folds improved substantially (mean of folds 0,1,2,4: **0.6696**)
- Fold 3 catastrophically failed: 0.3977 (below random chance), -0.21 vs baseline
- Fold 3 peaked at epoch 2 (still in warmup), never recovered — only 3 "real" epochs before early stop
- High variance (std 0.114 vs 0.020 baseline) makes the mean misleading
- Folds 0 and 4 exceeded 0.70 — strongest individual results in this project
- **Diagnosis**: Tuned config is too conservative for some splits. Warmup (2 epochs) + patience (3) = only 3 post-warmup chances. When a fold starts cold, it gets killed too early.
- **Next step**: Hybrid config — keep grad accumulation and dropout, but increase patience to 5 and reduce warmup to 1 epoch

---

---

## Experiment 3: BLCA Hybrid (never completed; superseded by the v2 protocol below)
**Date**: 2026-05-19
**Cancer type**: BLCA

### Config Changes (vs Tuned)
| Parameter | Tuned | Hybrid | Rationale |
|-----------|-------|--------|-----------|
| Warmup | 2 epochs | **1 epoch** | Less warmup = more real training before patience |
| Patience | 3 | **5** | Give slow folds room to recover (fold 3 collapsed in tuned) |
| All other params | same | same | LR=5e-5, dropout=0.25, grad_accum=8, weight_decay=1e-4 |

### Results
| Fold | C-Index | Peak Epoch | Stopped At | vs Baseline | vs Tuned |
|------|---------|------------|------------|-------------|----------|
| 0 | *running* | | | | |
| 1 | pending | | | | |
| 2 | pending | | | | |
| 3 | pending | | | | |
| 4 | pending | | | | |

### Key question
Does fold 3 recover with more patience and less warmup?

---

## Published Baselines (BLCA, DSS)
| Method | Venue | C-Index (approx) |
|--------|-------|-------------------|
| ABMIL (WSI only) | — | ~0.52-0.55 |
| MCAT | ICCV 2021 | ~0.58-0.61 |
| MOTCat | AAAI 2023 | ~0.59-0.62 |
| SurvPath | CVPR 2024 | ~0.60-0.63 |
| MMP | ICML 2024 | ~0.61-0.64 |

---

## Next Steps
- [ ] Complete tuned BLCA 5-fold CV
- [ ] Download embeddings for other 4 cancer types (BRCA, STAD, COADREAD, HNSC)
- [ ] Run tuned config on all 5 cancer types
- [ ] Ablation studies (vary K, modality dropout, etc.)
- [ ] Missing-modality experiments (Table 3 from research plan)
- [ ] Consider Karpathy's autoresearch for automating ablation sweeps
- [ ] Baseline reproduction (MCAT, SurvPath) for fair comparison

---

## Environment
- **GPU**: Google Colab Free (T4, 15GB VRAM)
- **WSI embeddings**: MahmoodLab/UNI2-h-features (HuggingFace, gated)
- **Data**: SurvPath repo splits, RNA-seq, pathway compositions
- **Framework**: PyTorch, lifelines (C-Index)

---

## 2026-09-10 - Protocol audit: every number above is superseded

A code review of the May pipeline found six defects that make Experiments 1-3 non-comparable
to each other and to published MCAT/SurvPath/MMP numbers:

| # | Defect | Effect | Fix (v2) |
|---|--------|--------|----------|
| 1 | C-index computed on the 4 **discrete bins**, not continuous time | ~75% of patient pairs tied and dropped; high variance; not the literature's metric | `sksurv.concordance_index_censored` on survival months |
| 2 | Bin edges fit **separately** on train and val (val quartiles) | val labels on a different scale than the model learned; leaks val label distribution | bins from uncensored **training** patients, outer edges +-inf, re-used for val |
| 3 | **Slide-level** samples (423) | multi-slide patients counted up to 9x; SurvPath is patient-level (359) | one sample per `case_id`, patches concatenated |
| 4 | Gene expression fed **unscaled** (log values -10..15) | pathway MLPs see arbitrary scales | MinMax to [-1,1] fit on train (SurvPath) |
| 5 | Selection **and** reporting on val C-index; no seeds | optimistic bias; fold-3 collapse unreproducible | selection on val loss (both logged); `seed + fold` |
| 6 | Risk = 1 - S_last; modality dropout per batch | non-standard risk; batch>1 semantics wrong | risk = -sum_t S_t; per-sample dropout, never both |

Architecture change made at the same time (configurable, `norm_first: true`): pre-norm query
blocks with a LayerNorm on the projected inputs. Old checkpoints cannot be loaded.

New evaluation for every run: IPCW C-index, Brier/IBS, td-AUC, 1000x bootstrap CI, KM log-rank,
and the missing-modality suite (WSI-only, genomics-only, 10-50% randomly missing) on the same
checkpoint. Everything is in `results.json` / `summary.md` per run; cross-run tables with paired
t / Wilcoxon tests via `scripts/aggregate_results.py`.

---

## Experiment 4: BLCA hybrid, v2 protocol
**Date**: 2026-09-10 | **Hardware**: local CPU (Ryzen AI MAX PRO 390, 12 cores) | **Config**: `configs/blca_hybrid_v2.yaml`
**Queue** (`scripts/run_experiments.sh`): hybrid -> WSI-only -> genomics-only -> baseline hyper-parameters,
then (chained by `scripts/chain_after.sh`) the Phase-6 baselines under the identical protocol:
official SurvPath, ABMIL (WSI only), SNN and MLP (genomics only) with the SurvPath-repo hyper-parameters.
Epoch time on this CPU: ~140 s for PathQ-Former (compute-bound, 11.4/12 cores busy).

**GPU queue (Runpod, launched 2026-09-10 ~15:00 local):** pod `pathq-gpu` (RTX PRO 4000 Blackwell, EU-RO-1, $0.57/h)
with network volume `pathq-embeddings` (250 GB). `scripts/pod/run_queue_pod.sh` streams each cohort's UNI2-h archive
into the volume and runs, under the identical protocol, PathQ-Former hybrid + SurvPath + ABMIL + SNN + MLP for
STAD, HNSC, COADREAD (COAD+READ archives) and BRCA (IDC+OTHERS archives), then BLCA seeds 1 and 2 for the two
multimodal methods. Results: `/workspace/outputs_v2/<run>/<cancer>/summary.md` and `/workspace/outputs_v2/summary_all.md`;
the pod removes itself when finished. Local hybrid v2 folds (BLCA seed 0): 0.600, 0.635, 0.701, 0.534, 0.754 -> 0.645 +- 0.086; 10/20/30/50 % randomly missing one modality: 0.647 / 0.639 / 0.621 / 0.623; optimistic best-epoch 0.669. Fold 3 selected epoch 2 (val loss) and is the weak fold again; fold 4 selected at 0.754 with only 16 events.

Results land in `outputs_v2/<run>/blca/summary.md`; consolidated table: `results/summary_all.md`.

**Fold 0 (finished 13:12, 26.7 min, early stop at epoch 11, selected epoch 6 by val loss):**
C-index **0.600** [bootstrap 95% CI 0.448, 0.720]; WSI-only 0.627; genomics-only 0.541;
10/20/30/50 % of patients missing one modality -> 0.605 / 0.595 / 0.582 / 0.609 (graceful).
Optimistic best-any-epoch 0.638, last epoch 0.619. Train loss 0.83 -> 0.33 after epoch 6 while val loss
rose: strong overfitting, which val-loss selection handles.

**Metric-grid decision (evidence from fold 0).** SurvPath's Brier/AUC grid includes val_min and val_max;
at val_max exactly one validation patient was still at risk, giving AUC 0.13 there versus 0.76 / 0.75 / 0.62
at the quartile edges and dragging the mean td-AUC to 0.37. Likewise the untruncated IPCW C-index was 0.46
because the training censoring survival G(t) is 0.03 at 120 months (weights 1/G^2 explode with 68 % censoring);
truncated at the 75th percentile of training event times (21 months) it is stable. v2 therefore evaluates
Brier/IBS/td-AUC on the interior quartile edges and Uno's C truncated at that 75th percentile
(`ipcw_tau` stored per fold). `scripts/recompute_metrics.py` re-derives all metric blocks from the saved
per-patient predictions, so runs finished before this change are made consistent; `scripts/finalize_after.sh`
does that automatically once both queues end. Harrell's C is unaffected.

Fill in here once the queue finishes:

| Run | C-index (mean +- std) | 95% CI | IPCW | IBS | td-AUC | WSI-only | Genomics-only |
|-----|----------------------|--------|------|-----|--------|----------|---------------|
| hybrid_v2 (seed 0, done 15:09) | **0.645 +- 0.086** | [0.538, 0.751] | 0.624 (tau ~21 mo) | 0.155 | 0.708 | 0.591 | 0.628 |
| baseline_v2 | | | | | | | |
| wsi_only | | | | | | n/a | n/a |
| genomic_only | | | | | | n/a | n/a |

---

## Experiment 5: multi-cohort on the GPU pod (val-loss protocol, seed 0) - finished 2026-09-11 05:26 local
Pod queue ran STAD, HNSC, BRCA (COADREAD crashed - see below) x {PathQ-Former hybrid v2, SurvPath, ABMIL, SNN, MLP}
plus BLCA seeds 1-2 for PathQ-Former and SurvPath. Full table: `pod_results/summary_all.md`.

| Cohort | PathQ-Former | SurvPath | ABMIL (WSI) | SNN (RNA) | MLP (RNA) |
|--------|-------------|----------|-------------|-----------|-----------|
| STAD | 0.486 +- 0.069 | **0.610 +- 0.074** | 0.498 | 0.454 | 0.516 |
| HNSC | 0.484 +- 0.084 | 0.515 +- 0.057 | **0.547** | 0.472 | 0.543 |
| BRCA | 0.548 +- 0.138 | 0.532 +- 0.184 | 0.481 | **0.572** | 0.539 |
| BLCA seed 1 | 0.599 +- 0.132 | 0.552 +- 0.142 | | | |
| BLCA seed 2 | 0.622 +- 0.105 | 0.597 +- 0.047 | | | |
| BLCA seed 0 (laptop) | 0.645 +- 0.086 | pending | | | |

**Reading.** Only BLCA carries signal (PathQ-Former 0.622 mean over 3 seeds vs SurvPath 0.575 over 2). On STAD, HNSC and
BRCA *every* method, including the official SurvPath and the plain RNA baselines, sits near chance, far below the
published 0.6-0.75 range. Training histories show the cause: with 7-16 events per validation fold the NLL is noise and
val-loss selection picked epoch 1-3 checkpoints while the validation C-index was still rising (e.g. STAD PathQ-Former
best-epoch mean 0.587 vs selected 0.486). PathQ-Former is hit hardest because lr 5e-5 with 8-step accumulation gives
~35 updates per epoch. This is a protocol failure, not an architecture result.

**Bugs found:** (1) COADREAD (37 events) crashed in the bootstrap CI - a resample with no comparable pairs makes sksurv
raise; `concordance()` now returns NaN there. (2) The pod could not remove itself (old runpodctl syntax); idle ~3 h.

**Decision (Experiment 6, queued 2026-09-11):** fixed-budget protocol `selection_metric: last` - train exactly 10 epochs and
report the final checkpoint, the MCAT/SurvPath convention - for PathQ-Former (lr 1e-4, accumulation 2:
`configs/protocol_fixed/pathq_fast_e10.yaml`) and SurvPath (`survpath_e10.yaml`) on all 5 cohorts, plus COADREAD under the
val-loss protocol for all 5 methods to complete Table 1 v1. Best-epoch numbers stay logged as an optimistic bound.

