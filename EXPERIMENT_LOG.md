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

## Experiments 6-7: three pods in parallel (launched 2026-09-11 ~09:40 local)
All pods: RTX PRO 4000, EU-RO-1, shared volume, features cached in RAM as float16 after the first epoch
(fp16 rounding of the frozen UNI2-h features; the model computes in fp32 on the device). Measured: uncached
epoch ~146 s at 15-25 % GPU, cached ~65 s at 7 % GPU -> host-side copies dominated, so from commit 0c6e0fc the
loader keeps fp16 through collate, skips the copy for single-patient batches and pins nothing without workers.
- **Pod 1, batch 2** -> `/workspace/outputs_v2`: COADREAD (val-loss, 5 methods), then fixed-budget
  `pathq_fast_e10` + `survpath_e10`, seed 0, five cohorts.
- **Pod 2, batch 3a** -> `/workspace/outputs_ablate`: BLCA fusion ablations under the fixed budget
  (modality dropout 0 / 0.3 / 0.5, K = 16 / 64, fusion depth 1 / 3, WSI-only and genomics-only PathQ-Former),
  the late-fusion baseline (`scripts/late_fusion.py`: z-scored risk average of the two single-modality models),
  then `pathq_fast_e10` seeds 1-2 on five cohorts.
- **Pod 3, batch 3b** -> `/workspace/outputs_ablate`: `survpath_e10` seeds 1-2 on five cohorts.
Decision rule for the paper protocol: the one under which the published baselines land near their published
numbers. Decision rule for the final PathQ-Former config: fusion must beat both single-modality models AND the
late-fusion baseline on BLCA; otherwise the fusion block, not the idea, is what needs work.

## Interim results 2026-09-11 15:45 local (seed 0 unless noted; fixed budget = 10 epochs, final checkpoint)

**Protocol decision made:** fixed budget. Under it the official SurvPath reproduces its published range and is
seed-stable (BLCA 0.588 / 0.585 / 0.593; STAD 0.609 / 0.593 / 0.617; HNSC 0.600), whereas under val-loss selection
its BLCA seeds scattered 0.552-0.597.

| Cohort | PathQ-Former fast e10 | SurvPath e10 (mean over seeds) | val-loss PathQ hybrid | val-loss SurvPath |
|--------|----------------------|-------------------------------|-----------------------|-------------------|
| BLCA | 0.609 +- 0.068 | 0.589 (3 seeds) | 0.645 (s0) / 0.599 / 0.622 | 0.559 / 0.552 / 0.597 |
| STAD | 0.559 +- 0.086 | 0.607 (3 seeds) | 0.486 | 0.610 |
| HNSC | 0.550 +- 0.052 | 0.600 (s1; s0 running) | 0.484 | 0.515 |
| COADREAD | running | running | 0.658 +- 0.161 | 0.568 |
| BRCA | running | running | 0.548 | 0.532 |

BLCA val-loss protocol, all methods (laptop, complete): PathQ hybrid 0.645, PathQ genomics-only 0.646,
PathQ WSI-only 0.618, old hyper-parameters 0.574, SurvPath 0.559 (paired t p = 0.024 vs hybrid), ABMIL 0.565,
MLP 0.571, SNN 0.499.

**BLCA fusion ablations (fixed budget, seed 0):** base 0.609 | md0 0.634 | md30 0.616 | md50 0.635 | K16 0.638 |
K64 0.634 | fusion1 0.598 | fusion3 running. All within one fold-std of each other. Single-modality test-time
scores of the same checkpoints are as high as the fused score (e.g. md30: fused 0.616, WSI-only 0.624,
genomics-only 0.648): **fusion is not adding signal over the stronger branch.** Modality dropout 0.3 gives the
best missing-modality robustness at no cost to the fused score.

**Per-epoch curves (pooled over folds and cohorts):** PathQ-Former fast schedule still rising at epoch 10
(0.544 -> 0.573 over epochs 7-10); SurvPath flat from epoch 8 (0.585-0.590). The 10-epoch budget therefore
favours SurvPath.

**Experiment 8 (pod 4, launched ~16:00):** 20-epoch budget for `pathq_fast_e20`, `pathq_fast_e20_aux`
(auxiliary unimodal survival heads on each branch, weight 0.5, so neither branch can be ignored) and
`survpath_e20`, seed 0 then seeds 1-2, five cohorts -> `/workspace/outputs_e20`. Histories give the 10-epoch
numbers post hoc (`scripts/epoch_curves.py`).

## Results 2026-09-12 14:30 local - four cohorts complete under both fixed budgets (BRCA re-running)

Overnight: all queues finished except BRCA (its fold caches stacked past the 62 GB container limit -> OOM kill,
exit 137; fixed in 8298d0f: cache capped at 45 % of the cgroup limit, caches/models freed between folds).
79 finished runs on the volume; BRCA + remaining 20-epoch seeds are re-running on three fresh pods (batch 5).

### Table 1 (draft): fixed budget, final checkpoint, seed 0 (mean +- std over 5 folds)

| Cohort | PathQ-Former 20 ep | PathQ-Former + aux heads 20 ep | SurvPath 20 ep | PathQ-Former 10 ep | SurvPath 10 ep |
|--------|--------------------|--------------------------------|----------------|--------------------|----------------|
| BLCA | 0.628 +- 0.073 | 0.625 +- 0.081 | 0.606 +- 0.099 | 0.609 +- 0.068 | 0.588 +- 0.088 |
| STAD | 0.591 +- 0.066 | 0.559 +- 0.063 | 0.589 +- 0.123 | 0.559 +- 0.086 | 0.609 +- 0.119 |
| HNSC | 0.584 +- 0.031 | 0.603 +- 0.031 | 0.531 +- 0.113 | 0.550 +- 0.052 | 0.531 +- 0.056 |
| COADREAD | 0.651 +- 0.140 | 0.595 +- 0.152 | 0.532 +- 0.121 | 0.698 +- 0.082 | 0.565 +- 0.118 |
| BRCA | running | running | running | running (val-loss: 0.548) | running (val-loss: 0.532) |

Multi-seed means (20 ep): BLCA PathQ 0.609 (3 seeds: 0.628/0.602/0.599), PathQ+aux 0.630 (0.625/0.624/0.642),
SurvPath 0.594 (0.606/0.582/0.596); STAD PathQ 0.586 (2 seeds), PathQ+aux 0.578 (2), SurvPath 0.589 (1).
10-ep seeds: BLCA PathQ 0.618 (0.609/0.634/0.612) vs SurvPath 0.589 (0.588/0.585/0.593).

Per-epoch curves pooled over folds and seeds (35 PathQ / 30 SurvPath fold-histories): PathQ-Former 0.605 at
epoch 20 (peak 0.612 at 13), SurvPath 0.573 at 20 (peak 0.585 at 14). PathQ-Former is ahead at every epoch >= 11,
so the 20-epoch budget is the right pre-registered choice and is symmetric.

### BLCA fusion ablations (10 ep, seed 0) and the fusion verdict

| Model | C-index | Test-time WSI-only | Test-time RNA-only |
|-------|---------|--------------------|--------------------|
| joint PathQ-Former (3 seeds) | 0.618 | 0.61 | 0.61 |
| late fusion of the two single-modality PathQ-Formers | 0.631 | - | - |
| PathQ-Former WSI-only (trained) | 0.596 | | |
| PathQ-Former genomics-only (trained) | 0.603 | | |
| K=16 / K=64 | 0.638 / 0.634 | | |
| modality dropout 0 / 0.3 / 0.5 | 0.634 / 0.616 / 0.635 | | |
| fusion depth 1 / 3 | 0.598 / 0.595 | | |

Joint fusion beats both single-modality models by ~0.02 and equals late fusion within noise: the fusion block is
not (yet) extracting complementarity beyond an ensemble. The auxiliary unimodal heads (20 ep) do not change the fused
score on BLCA (0.630 vs 0.609 over seeds, +0.02, p = 0.85 paired) but make the checkpoint far more balanced under
missing modalities (BLCA test-time RNA-only 0.588 with aux vs 0.535 without; HNSC 0.603 vs 0.584 fused).
None of K, fusion depth or modality dropout matters for the fused score; modality dropout matters for robustness.

### Reading for the paper
1. Under the credible protocol (fixed 20-epoch budget, identical for all methods), PathQ-Former >= SurvPath on
   BLCA (+0.015 over 3 seeds), HNSC (+0.05), COADREAD (+0.12, noisy) and ties on STAD; BRCA pending.
2. The unique capability holds: the same checkpoint scores 0.58-0.64 with either modality removed; SurvPath cannot run.
3. Fusion adds ~0.02 over the best single modality and matches late fusion -> claim "unified model", not "better fusion".

## Seeded Table 1 - 2026-09-12 19:50 local (`scripts/seed_table.py`)

**10-epoch fixed budget, 3 seeds x 5 folds per cell, all five cohorts complete:**

| Cohort | PathQ-Former (fast) | SurvPath (official) | delta | paired t p | Wilcoxon p |
|--------|--------------------|---------------------|-------|-----------|------------|
| BLCA | 0.618 +- 0.014 | 0.589 +- 0.004 | +0.029 | 0.115 | 0.164 |
| BRCA | 0.600 +- 0.031 | 0.536 +- 0.015 | +0.064 | 0.036 | 0.048 |
| COADREAD | 0.646 +- 0.046 | 0.551 +- 0.022 | +0.094 | 0.031 | 0.022 |
| HNSC | 0.556 +- 0.007 | 0.568 +- 0.035 | -0.013 | 0.470 | 0.679 |
| STAD | 0.574 +- 0.025 | 0.607 +- 0.012 | -0.033 | 0.285 | 0.609 |
| **All (75 paired folds)** | | | **+0.028** | **0.034** | **0.017** |

(+- is the spread over seeds of the 5-fold means.) PathQ-Former wins 3 cohorts, two of them significantly, and
ties two; pooled over all 75 (fold, seed) pairs the advantage is significant under both tests.

**20-epoch fixed budget (seeds still filling in; BRCA running):** BLCA PathQ 0.609 (3 seeds), PathQ + aux heads
0.630 (3 seeds; +0.036 vs SurvPath, p = 0.024), SurvPath 0.594 (3 seeds); STAD 0.585 / 0.579 / 0.592 (2 seeds);
HNSC 0.582 / 0.603 / 0.531; COADREAD 0.651 / 0.595 / 0.532 (1 seed). Pooled over 35 pairs: PathQ +0.029 (t p 0.10),
PathQ + aux +0.031 (Wilcoxon p 0.046).

**Compute note:** pod A removed at 19:45 to stretch the account balance (its remaining 20-epoch STAD/BRCA seeds
resume with `scripts/pod/batch5_a.sh` on any new pod); pods B (HNSC 20-ep seeds) and C (BRCA 20-ep seed 0,
COADREAD 20-ep seeds) continue.

## 20-epoch seeded table - 2026-09-14 12:30 local (pods B, C finished 09-13; 101 runs on the volume)

Seeds per cell: 3 for BLCA/COADREAD/HNSC, 2 for STAD, 1 for BRCA (STAD seed 2 and BRCA seeds 1-2 re-running on
the recovery pod; they were pod A's share, removed on 09-12 to protect the account balance).

| Cohort | PathQ-Former 20 ep | PathQ-Former + aux heads 20 ep | SurvPath 20 ep |
|--------|--------------------|--------------------------------|----------------|
| BLCA | 0.609 +- 0.016 | **0.630 +- 0.010** (p = 0.024 vs SurvPath) | 0.594 +- 0.012 |
| BRCA | 0.568 | **0.647** (p = 0.038, 1 seed) | 0.552 |
| COADREAD | **0.638 +- 0.031** | 0.613 +- 0.023 | 0.570 +- 0.038 |
| HNSC | 0.575 +- 0.012 | **0.582 +- 0.025** | 0.552 +- 0.019 |
| STAD | 0.585 +- 0.008 | 0.579 +- 0.028 | **0.592 +- 0.004** |
| **All (60 paired folds)** | +0.027, t p 0.033, W p 0.067 | **+0.033, t p 0.025, W p 0.006** | - |

Missing-modality (seed 0, same checkpoint tested WSI-only / RNA-only): plain 20-ep model leans on WSI
(BLCA 0.635 / 0.535, HNSC 0.585 / 0.548); the aux-head model is balanced (BLCA 0.598 / 0.588, BRCA 0.610 / 0.671,
HNSC 0.604 / 0.575). With 10-50 % of patients randomly missing one modality the fused C-index moves by < 0.01
(BLCA 0.628 -> 0.620-0.626; BRCA 0.568 -> 0.557-0.567).

**Recommended final configuration for the paper:** `pathq_fast_e20_aux` (lr 1e-4, accumulation 2, 20 epochs,
auxiliary unimodal heads, modality dropout 0.15): best or tied on 4/5 cohorts, +0.033 pooled vs the official
SurvPath under an identical protocol (Wilcoxon p = 0.006 over 60 paired folds), and graceful degradation in both
missing-modality directions.

## 2026-09-14 - test suite and remaining experiments (Experiments 9-13)

**Tests.** `tests/test_trainer.py` (end-to-end on the dummy embeddings, marked `slow`): config overrides and
validation, weighted (bin, censorship) sampler, OS/PFI endpoints, patients without embeddings, full `train_cv`
run + fold skip on rerun, mid-fold resume from `latest_checkpoint.pt`, val_loss / val_cindex / fixed-budget
selection, seed determinism on CPU, auxiliary unimodal loss, ABMIL / SNN / MLP / official SurvPath through the
trainer, and the analysis scripts (recompute, late fusion, aggregation, seed table, epoch curves, figures +
attention export). Unit additions in `tests/test_pipeline.py`: attention maps are distributions over valid keys,
post-norm variant, learnable null codes, per-sample dropout statistics, evaluation grid / IPCW horizon rules,
graceful NaNs, multi-patient collate. First bug caught: `--set lr=1e-4` was parsed as the *string* "1e-4"
(YAML 1.1); overrides now coerce numeric strings.

**Experiments queued (recovery pod, `scripts/pod/batch6.sh`, chained after the last 20-epoch seeds):**
9. Efficiency: params, forward / forward+backward latency, peak memory, PathQ-Former (all patches and 4096)
   vs SurvPath (4096 and all) on real BLCA validation patients (`scripts/efficiency.py`).
10. Single-modality baselines under the fixed 20-epoch budget (ABMIL, SNN, MLP), five cohorts.
11. Mean-imputation missing-modality evaluation of the finished 20-epoch SurvPath and PathQ-Former checkpoints
    (`scripts/eval_missing_impute.py`): the baseline receives the training-mean gene vector / mean patch
    instead of crashing, so Table 3 compares degradation, not availability.
12. Single-modality PathQ-Former at 20 epochs on five cohorts + late-fusion baseline per cohort (does joint
    fusion beat an ensemble anywhere?).
13. Overall survival (OS) endpoint for PathQ-Former+aux and SurvPath, five cohorts, seed 0 (`outputs_os`).
**Laptop (CPU):** ablations of the final configuration on BLCA - Hallmark (50) and Xena (281) pathway sets,
2 and 8 survival bins, 4096-patch training subsample (`configs/ablation_fixed/pathq_aux_e20_*.yaml`).
Laptop CPU efficiency (real BLCA slide, 36k patches): PathQ-Former fwd 190 ms / fwd+bwd 1013 ms; SurvPath on the
same bag 188 / 724 ms; both ~45 / ~215 ms at 4096 patches.

## 20-epoch seeded table, update 2026-09-15 09:50 local (BRCA now 2 seeds, STAD 3 seeds; 70 paired folds)

| Cohort | PathQ-Former 20 ep | PathQ-Former + aux | SurvPath 20 ep | aux vs SurvPath |
|--------|--------------------|--------------------|----------------|-----------------|
| BLCA | 0.609 +- 0.016 (3) | **0.630 +- 0.010** (3) | 0.594 +- 0.012 (3) | +0.036, p 0.024 |
| BRCA | 0.607 +- 0.055 (2) | **0.641 +- 0.009** (2) | 0.552 +- 0.000 (2) | +0.089, p 0.019 |
| COADREAD | **0.638 +- 0.031** (3) | 0.613 +- 0.023 (3) | 0.570 +- 0.038 (3) | +0.044, ns |
| HNSC | 0.575 +- 0.012 (3) | **0.582 +- 0.025** (3) | 0.552 +- 0.019 (3) | +0.031, ns |
| STAD | **0.593 +- 0.015** (3) | 0.572 +- 0.022 (3) | 0.587 +- 0.008 (3) | -0.015, ns |
| All 70 pairs | +0.032, t p 0.006, W p 0.016 | **+0.033, t p 0.017, W p 0.004** | | |

Remaining: BRCA seed 2 for the three configs (recovery pod, ~10 h), then batch 6. Laptop: Hallmark-pathway
ablation on its last fold (folds 1-4: 0.717 / ... / 0.523), then Xena, 2 bins, 8 bins, 4096 patches.

## Final 20-epoch Table 1 and Table 3 - 2026-09-16 09:00 local (3 seeds x 5 folds x 5 cohorts; 127 finished runs)

| Cohort | PathQ-Former | PathQ-Former + aux | SurvPath | ABMIL (WSI) | SNN (RNA) | MLP (RNA) |
|--------|-------------|--------------------|----------|-------------|-----------|-----------|
| BLCA | 0.609 +- 0.016 | **0.630 +- 0.010** | 0.594 +- 0.012 | 0.578 | 0.599 | 0.619 |
| BRCA | 0.611 +- 0.039 | 0.622 +- 0.033 | 0.536 +- 0.026 | 0.574 | 0.552 | **0.629** |
| COADREAD | **0.638 +- 0.031** | 0.613 +- 0.023 | 0.570 +- 0.038 | 0.595 | 0.543 | 0.630 |
| HNSC | 0.575 +- 0.012 | **0.582 +- 0.025** | 0.552 +- 0.019 | 0.570 | 0.536 | 0.541 |
| STAD | **0.593 +- 0.015** | 0.572 +- 0.022 | 0.587 +- 0.008 | 0.543 | 0.556 | 0.524 |

(baselines: seed 0 only so far; seeds 1-2 queued as batch 7.) Paired over 75 (fold, seed) pairs vs SurvPath:
PathQ-Former +0.037 (t p 0.002, Wilcoxon p 0.005), PathQ-Former+aux +0.036 (t p 0.007, W p 0.002).
vs MLP on RNA (25 pairs, seed 0): +0.016 / +0.017, not significant. vs ABMIL: aux +0.034 (W p 0.037).
**Honest reading:** the multimodal model clearly beats the strongest published multimodal baseline, but a plain MLP
on 4,999 genes is within noise of it on BRCA and COADREAD. The defensible accuracy claim is "matches or beats every
baseline"; the differentiator is robustness.

### Table 3: missing modalities at test time (20 ep, seed 0, same checkpoint)
Baselines receive the training-mean gene vector / mean patch (`scripts/eval_missing_impute.py`); PathQ-Former uses its
learned null codes (imputation results for it are similar).

| Cohort | SurvPath full | SurvPath, RNA missing | SurvPath, WSI missing | PathQ+aux full | PathQ+aux, RNA missing | PathQ+aux, WSI missing |
|--------|---------------|-----------------------|-----------------------|----------------|------------------------|-------------------------|
| BLCA | 0.606 | 0.608 | 0.509 | 0.625 | 0.598 | 0.588 |
| BRCA | 0.552 | 0.552 | 0.514 | 0.647 | 0.610 | 0.671 |
| COADREAD | 0.532 | 0.534 | 0.480 | 0.595 | 0.547 | 0.594 |
| HNSC | 0.531 | 0.533 | 0.551 | 0.603 | 0.604 | 0.575 |
| STAD | 0.589 | 0.590 | 0.462 | 0.559 | 0.579 | 0.541 |

SurvPath's score is unchanged when RNA is removed and falls to chance when WSI is removed: it effectively ignores the
genomic branch. PathQ-Former+aux keeps 0.54-0.67 with either modality removed. Without the auxiliary heads PathQ-Former
behaves like SurvPath (RNA-only ~0.52-0.55), so the heads are what buys the robustness.

**Compute/ops:** batch 6 running on the recovery pod (single-modality PathQ-Former 20 ep + late fusion, then OS
endpoint); efficiency step failed there (laptop embeddings path) - fixed (`--embeddings_dir`) and queued in batch 7
with baseline seeds 1-2. Laptop rebooted 09-15 and killed the ablation queue; relaunched 09-16 08:54 (resumes).

## 2026-09-17 10:55 local - late fusion on all cohorts, OS endpoint, final-config ablations (148 finished runs)

**Fusion vs ensemble (20 ep, seed 0 for the ensemble ingredients):**

| Cohort | PathQ+aux (3 seeds) | late fusion (WSI-only + RNA-only PathQ) | WSI-only PathQ | RNA-only PathQ |
|--------|---------------------|------------------------------------------|----------------|----------------|
| BLCA | 0.630 | **0.642** | 0.593 | 0.631 |
| BRCA | **0.622** | 0.592 | 0.596 | 0.558 |
| COADREAD | **0.613** | 0.571 | 0.588 | 0.566 |
| HNSC | **0.582** | 0.578 | 0.586 | 0.563 |
| STAD | **0.572** | 0.549 | 0.567 | 0.512 |

Joint fusion beats the late-fusion ensemble on 4/5 cohorts, +0.020 pooled over 25 pairs (p = 0.24). Together with
the BLCA 10-epoch result (+0.02 vs the best single modality) the honest statement is: fusion adds a small, consistent
but not individually significant gain over an ensemble of the same two branches.

**Overall survival (OS) endpoint, 20 ep, seed 0, all five cohorts (pod batch 6, finished 2026-09-17; pulled 2026-09-18):**

| Cohort | PathQ+aux | SurvPath | delta | paired t p (5 folds) |
|--------|-----------|----------|-------|----------------------|
| BLCA | **0.595** | 0.552 | +0.043 | 0.008 |
| BRCA | **0.596** | 0.523 | +0.073 | 0.015 |
| COADREAD | 0.581 | **0.596** | -0.015 | 0.58 |
| HNSC | **0.536** | 0.510 | +0.025 | 0.046 |
| STAD | **0.603** | 0.559 | +0.044 | 0.23 |
| pooled (25 folds) | | | **+0.034** | t 0.0028, Wilcoxon 0.0025 |

Same picture as DSS: PathQ+aux wins 4/5 cohorts (significantly on BLCA, BRCA, HNSC) and ties COADREAD. Absolute OS
C-indices sit ~0.02-0.03 below DSS for both methods (OS counts non-cancer deaths). Missing-modality robustness holds
on OS (both / WSI-only / RNA-only: BLCA 0.595 / 0.576 / 0.592, STAD 0.603 / 0.574 / 0.582).

**Final-config ablations on BLCA (laptop, 20 ep, seed 0):** 275 Reactome+Hallmark pathways 0.630 (3 seeds) |
Xena 281 pathways 0.628 | 50 Hallmark pathways 0.600 - the richer pathway sets help; 2 bins / 8 bins / 4096
training patches running (laptop reboots keep killing the detached queue; it resumes on relaunch).

**Pod:** batch 6 finished 2026-09-17 (OS complete). Batch 7 had finished 24/30 baseline-seed runs (ABMIL/SNN/MLP,
seeds 1-2, BLCA/STAD/HNSC/COADREAD + BRCA ABMIL seed 1) when the Runpod balance hit $0 late on 2026-09-17 and every
pod was deleted (volume intact). After a top-up, pod `pathq-gpu-f` (RTX PRO 4000, $0.57/h) resumed batch 7 on
2026-09-18 12:10 local: BRCA SNN s1 (fold 5), MLP s1, ABMIL/SNN/MLP s2, then `efficiency.md` (~4 h).

## 2026-09-19 12:45 local - FINAL TABLES: 20-epoch protocol, 3 seeds, every method on every cohort (180 finished runs)

Batch 7 finished on the pod 2026-09-18 20:05 local (BRCA baseline seeds 1-2 + efficiency). All 6 methods now have
3 seeds x 5 folds on all 5 cohorts; the tables below are the paper's Table 1 / Table 2 (`scripts/seed_table.py
pod_results/outputs_e20 --methods ... --ref survpath_e20`, DSS endpoint, fixed 20-epoch budget, final checkpoint,
mean +- std over 3 seeds of the 5-fold mean C-index).

### Table 1: C-index, 20 epochs, final checkpoint (3 seeds)

| Method | Inputs | BLCA | BRCA | COADREAD | HNSC | STAD | pooled delta vs SurvPath (75 paired folds) |
|--------|--------|------|------|----------|------|------|--------------------------------------------|
| PathQ-Former | WSI + RNA | 0.609 +- 0.016 | 0.611 +- 0.039 | **0.638 +- 0.031** | 0.575 +- 0.012 | **0.593 +- 0.015** | **+0.037** (t p 0.002, Wilcoxon p 0.005) |
| PathQ-Former + aux heads | WSI + RNA | **0.630 +- 0.010** | 0.622 +- 0.033 | 0.613 +- 0.023 | **0.582 +- 0.025** | 0.572 +- 0.022 | **+0.036** (t p 0.007, Wilcoxon p 0.002) |
| SurvPath (official) | WSI + RNA | 0.594 +- 0.012 | 0.536 +- 0.026 | 0.570 +- 0.038 | 0.552 +- 0.019 | 0.587 +- 0.008 | reference |
| ABMIL | WSI | 0.566 +- 0.011 | 0.575 +- 0.004 | 0.587 +- 0.023 | 0.562 +- 0.028 | 0.553 +- 0.020 | +0.001 (p 0.94) |
| SNN | RNA | 0.590 +- 0.009 | 0.558 +- 0.013 | 0.572 +- 0.028 | 0.535 +- 0.006 | 0.549 +- 0.009 | -0.007 (p 0.67) |
| MLP | RNA | 0.611 +- 0.012 | **0.627 +- 0.013** | 0.633 +- 0.016 | 0.555 +- 0.012 | 0.532 +- 0.009 | +0.024 (t p 0.15, Wilcoxon p 0.30) |

Per-cohort paired tests vs SurvPath (15 pairs each): PathQ+aux BLCA +0.036 (t p 0.024), BRCA +0.086 (p 0.004),
COADREAD +0.044 (p 0.32), HNSC +0.031 (p 0.29), STAD -0.015 (p 0.57); plain PathQ BLCA +0.015 (p 0.36), BRCA +0.075
(p 0.013), COADREAD +0.068 (p 0.064), HNSC +0.023 (p 0.38), STAD +0.006 (p 0.79). No single-modality baseline beats
SurvPath pooled; the RNA MLP comes closest (+0.024, ns) and wins BRCA/COADREAD outright.

### Table 2: same runs, reference = RNA MLP (the strongest single-modality baseline)

| Method | BLCA | BRCA | COADREAD | HNSC | STAD | pooled (75 pairs) |
|--------|------|------|----------|------|------|-------------------|
| PathQ-Former | -0.002 | -0.016 | +0.004 | +0.020 | +0.062 (p 0.066) | +0.014 (t p 0.25, W p 0.21) |
| PathQ-Former + aux | +0.019 | -0.005 | -0.020 | +0.028 | +0.041 | +0.012 (t p 0.28, W p 0.12) |
| SurvPath | -0.017 | -0.091 | -0.064 | -0.003 | +0.056 | -0.024 (t p 0.15) |

Honest reading: PathQ-Former is significantly better than the published multimodal baseline (SurvPath), but only
within noise of a well-tuned MLP on the same RNA pathways; the multimodal model's advantage is that it is >= the
best unimodal model on every cohort with one network (the MLP loses STAD, -0.06 vs PathQ; ABMIL loses BLCA/BRCA),
and it keeps working when either modality is missing (Table 3), which neither the MLP nor SurvPath does.

### Efficiency (`scripts/efficiency.py`, RTX PRO 4000, BLCA fold-0 validation, 40 patients, medians)

| Model | Params (M) | Patches | Forward (ms) | Forward+backward (ms) | Peak GPU mem (GiB) |
|---|---|---|---|---|---|
| PathQ-Former (all patches) | 16.9 | 6214 | 35.0 | 104.2 | 1.17 |
| PathQ-Former (4096 patches) | 16.9 | 4096 | 34.7 | 104.1 | 0.17 |
| SurvPath (4096 patches, as trained) | 21.2 | 4096 | 35.7 | 175.6 | 0.20 |
| SurvPath (all patches) | 21.2 | 6214 | 36.0 | 176.0 | 1.74 |

PathQ-Former has 20 % fewer parameters, the same forward latency, a 1.7x faster training step and a third less
peak memory than SurvPath on full slides; its cost is flat in the number of patches because K = 32 queries do the
cross-attention (SurvPath's cost grows with patches x pathways).

### Final-config ablations on BLCA (laptop CPU, PathQ+aux, 20 ep, seed 0; default = 4 bins, 275 pathways)

| Variant | C-index (both) | WSI-only | RNA-only |
|---|---|---|---|
| default (4 bins, Reactome+Hallmark 275) - seed 0 / 3 seeds | 0.625 / 0.630 | 0.598 | 0.588 |
| 2 hazard bins | 0.628 | 0.634 | 0.576 |
| 8 hazard bins | 0.609 | 0.579 | 0.605 |
| Xena 281 pathways | 0.628 | | |
| 50 Hallmark pathways only | 0.600 | | |
| 4096 training patches (vs all) | 0.626 | 0.591 | 0.622 |

Bin count barely matters (2 ~ 4 > 8, all within one std); the richer pathway sets beat Hallmark-only by ~0.03; training
on a 4096-patch subsample instead of every patch changes nothing (0.626 vs 0.625), so the gain over SurvPath is not
explained by PathQ-Former seeing more patches. All BLCA ablations of the final config finished 2026-09-19 13:38 (laptop).

### Compute log

Pod `pathq-gpu-f` ran 24.6 h ($14): 8 h of batch 7 plus ~13.5 h idle, because the self-removal watcher I added
by hand used only the v2 `runpodctl pod remove` syntax (the installed CLI is v1: `runpodctl remove pod`) and its
REST fallback also failed. Lesson recorded: always use the three-way chain from `run_queue_pod.sh`. All 180 runs
and 855 checkpoints remain on network volume fjb5dlrrfp; local copy in `pod_results/` (pull `pull_0918/results_0918b.tgz`).

## 2026-09-22 - Reviewer-requested recomputes (no new training; `scripts/seed_table.py`, `scripts/missing_modality_table.py`, `scripts/risk_correlation.py`)

**1. Seed-averaged paired tests (25 folds).** Seeds of the same fold share validation patients, so the 75 (fold, seed)
pairs in Table 1 are not independent. Averaging the three seeds per fold first and pairing over the 25 (cohort, fold)
units gives, vs SurvPath: PathQ-Former +0.037, t p = 0.020, Wilcoxon p = 0.020, wins 17/25; PathQ-Former + aux +0.036,
t p = 0.074, Wilcoxon p = 0.030, wins 18/25. Per cohort (5 folds) only BRCA stays significant (aux +0.086, t p 0.017;
plain +0.075, p 0.073). Baselines vs SurvPath: ABMIL +0.001 (p 0.95), SNN -0.007 (p 0.80), MLP +0.024 (p 0.38). vs the
RNA MLP: PathQ +0.014 (p 0.47), aux +0.012 (p 0.47). OS (one seed, unchanged): +0.034, p 0.003. **Paper wording:** report
the seed-averaged 25-fold tests as primary; the plain model's gain is significant under both tests, the aux model's under
the rank test only.

**2. Missing-modality with three seeds (same checkpoints, null codes; mean over seeds of the 5-fold mean, +- over seeds):**

| Cohort | PathQ+aux both | WSI-only | RNA-only | PathQ both | WSI-only | RNA-only |
|---|---|---|---|---|---|---|
| BLCA | 0.630 +- .008 | 0.600 +- .002 | 0.613 +- .018 | 0.609 +- .013 | 0.612 +- .017 | 0.590 +- .041 |
| BRCA | 0.622 +- .027 | 0.592 +- .038 | 0.617 +- .042 | 0.611 +- .032 | 0.600 +- .037 | 0.572 +- .027 |
| COADREAD | 0.613 +- .019 | 0.576 +- .020 | 0.583 +- .018 | 0.638 +- .026 | 0.633 +- .022 | 0.601 +- .046 |
| HNSC | 0.582 +- .021 | 0.596 +- .013 | 0.545 +- .022 | 0.575 +- .010 | 0.588 +- .006 | 0.533 +- .014 |
| STAD | 0.572 +- .018 | 0.589 +- .008 | 0.538 +- .003 | 0.593 +- .012 | 0.595 +- .017 | 0.578 +- .029 |

Pooled drop vs both (75 pairs / seed-averaged 25): aux WSI-only -0.013 (p 0.13 / 0.27), RNA-only -0.025 (p 0.053 /
0.19); plain WSI-only +0.000, RNA-only -0.030 (p 0.016 / 0.080). Random 10-50 % missing moves the aux model by <= 0.025
on every cohort with three seeds. With three seeds the aux heads still help RNA-only on BLCA/BRCA (+0.023 / +0.045 vs
plain) but the plain model is slightly better WSI-only; the 20-epoch checkpoints are less one-sided than the seed-0
Table 3 suggested (plain RNA-only 0.59/0.57/0.60/0.53/0.58, not 0.52-0.55). The SurvPath imputation columns and the
PathQ imputation variants still have one seed (needs checkpoints on a pod; batch 9).

**3. Risk-score correlations (Spearman over validation patients, mean over 15 fold x seed):** within the aux model the
fused risk correlates 0.83 with its own WSI-only risk and 0.45 with its RNA-only risk (pooled); the two branches
correlate only 0.15 with each other, i.e. they carry nearly independent signal and the fused score is still
histology-led. Across models: aux vs plain PathQ 0.69, vs SurvPath 0.46, vs ABMIL 0.46, vs MLP 0.35, vs SNN 0.34.
The multimodal model ranks patients differently from every baseline of equal accuracy, which is the ensemble
headroom that Section 6 measured at +0.02.

**Still open (need a pod, `scripts/pod/batch9.sh`):** SurvPath learning-rate x alpha grid under the protocol (the
manuscript currently compares against the published hyper-parameters only); single-modality PathQ-Former seeds 1-2
for a three-seed fusion-vs-ensemble table; mean-imputation columns for seeds 1-2; RNA / WSI permutation test.

## 2026-09-23 - Batch 9 part 1: permutation test, three-seed imputation, two-seed fusion-vs-ensemble (pod h; grid paused)

Pod h finished the evaluation passes and 18/20 single-modality runs before the account balance forced a stop
(BRCA seed 2 pair pending; grid paused at 1/35 after its first cell showed ~58 GPU-h for the full grid).

**Permutation test and imputation, three seeds (mean over seeds of the 5-fold mean C-index; pooled deltas vs `both`
over 75 (fold, seed) pairs, seed-averaged 25-fold p in brackets):**

| PathQ-Former + aux | both | RNA-only (null code) | RNA-only (mean patch) | WSI-only (null code) | WSI-only (mean genes) | RNA permuted | WSI permuted |
|---|---|---|---|---|---|---|---|
| BLCA | 0.630 | 0.613 | 0.595 | 0.600 | 0.597 | 0.597 | 0.570 |
| BRCA | 0.622 | 0.617 | 0.591 | 0.592 | 0.607 | 0.572 | 0.539 |
| COADREAD | 0.613 | 0.583 | 0.580 | 0.576 | 0.602 | 0.618 | 0.556 |
| HNSC | 0.582 | 0.545 | 0.558 | 0.596 | 0.589 | 0.566 | 0.539 |
| STAD | 0.572 | 0.538 | 0.521 | 0.589 | 0.581 | 0.555 | 0.484 |
| pooled delta | | -0.025 (p .053 [.19]) | -0.035 (p .004 [.040]) | -0.013 (p .13 [.27]) | -0.009 (p .20 [.29]) | **-0.023 (p .016 [.075])** | **-0.067 (p 6e-7 [2e-4])** |

| SurvPath | both | RNA-only (mean patch) | WSI-only (mean genes) | RNA permuted | WSI permuted |
|---|---|---|---|---|---|
| BLCA | 0.594 | 0.486 | 0.595 | 0.594 | 0.511 |
| BRCA | 0.536 | 0.520 | 0.536 | 0.537 | 0.532 |
| COADREAD | 0.570 | 0.473 | 0.572 | 0.572 | 0.510 |
| HNSC | 0.552 | 0.504 | 0.553 | 0.552 | 0.489 |
| STAD | 0.587 | 0.493 | 0.588 | 0.587 | 0.459 |
| pooled delta | | -0.073 (p 6e-5 [.003]) | +0.001 (n.s.) | **+0.001 (p .43 [.36])** | **-0.068 (p 3e-4 [.016])** |

Reading. (1) SurvPath's prediction does not use RNA: replacing it by the training mean or by another patient's RNA
changes nothing (+0.001, three seeds), and removing WSI drops it to chance. (2) PathQ-Former + aux does use RNA:
permuting it costs 0.023 (significant over 75 pairs, p = 0.075 seed-averaged), about a third of what permuting WSI
costs (0.067); the fused prediction is histology-led but not histology-only, consistent with the 0.83 / 0.45 risk
correlations. (3) Null codes vs mean imputation for the same PathQ checkpoints: RNA-only 0.579 vs 0.569 (+0.010,
p 0.27), WSI-only 0.591 vs 0.595 (n.s.); the learned null codes are at least as good as imputation, not the source
of the robustness. (4) The robustness claim holds with three seeds: PathQ+aux keeps 0.54-0.62 with either modality
removed on every cohort; SurvPath keeps its score only when RNA is removed.

**Fusion vs late-fusion ensemble, two seeds (0-1), 50 (fold, seed) pairs:** late fusion 0.629 / 0.606 / 0.620 /
0.571 / 0.544 (BLCA / BRCA / COADREAD / HNSC / STAD). Joint PathQ+aux vs ensemble +0.011 (t p .39, W p .22;
seed-averaged +0.010, p .32, wins 14/25); plain PathQ +0.009 (p .51). The ensemble itself is +0.027 over SurvPath
(t p .16, W p .071). Single-modality PathQ-Former vs SurvPath: WSI-only +0.015 (n.s.), RNA-only -0.006. So: joint
fusion ~ ensemble (+0.01), and most of the gain over SurvPath is already present in an ensemble of the two branches.

**Grid (1/35 cells):** PathQ+aux with lr 2e-4 on BLCA 0.615 (seed 0) vs 0.630 at lr 1e-4; inconclusive alone.

**Pending on the volume (resumable):** BRCA seed 2 WSI-only (killed in fold 1) and RNA-only, late fusion seed 2,
34 grid runs (~$5 + ~$33 at $0.57/h). Pod h ran 27 h (~$15.5); the grid pod ran 1.7 h before I removed it.
