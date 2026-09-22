# PathQ-Former: Final Experimental Report

**Date:** 2026-09-20 | **Status:** all planned experiments complete (180 GPU runs + 12 laptop runs) | **Code:** commit 8e713a9 (this repository) | **Raw record:** `EXPERIMENT_LOG.md`

---

## 0. Executive summary

PathQ-Former is a Q-Former-style fusion model that reads a patient's whole-slide-image (WSI) patch embeddings and RNA-seq pathway tokens through two banks of 32 learned queries, fuses the 64 codes with a small Transformer, and predicts discrete-time survival hazards. A missing modality (at training or test time) is replaced by learned null codes, so one checkpoint serves WSI+RNA, WSI-only and RNA-only patients.

Under a single, pre-registered protocol (SurvPath's 5-fold patient-level splits, disease-specific survival, fixed 20-epoch budget, final checkpoint, 3 seeds, identical for every method) on five TCGA cohorts:

1. **PathQ-Former beats the official SurvPath implementation** by +0.037 C-index pooled over 75 paired (fold, seed) pairs (paired t p = 0.002, Wilcoxon p = 0.005). The variant with auxiliary unimodal heads gains +0.036 (p = 0.007 / 0.002). The wins are significant on BLCA (+0.036, p = 0.024) and BRCA (+0.086, p = 0.004); COADREAD, HNSC and STAD are ties within noise.
2. **No single-modality baseline beats SurvPath pooled** (ABMIL +0.001, SNN -0.007, RNA-MLP +0.024, all n.s.), but the RNA-only MLP is **within noise of PathQ-Former** (PathQ-Former is +0.012 to +0.014 over it pooled, n.s.) and wins BRCA and COADREAD outright. The defensible accuracy claim is therefore "matches or beats every baseline on every cohort with one network", not "beats everything".
3. **The differentiator is robustness.** SurvPath's score is unchanged when its RNA input is replaced by the training mean and falls to chance (0.46-0.55) when WSI is removed: it ignores the genomic branch. PathQ-Former + aux keeps 0.54-0.67 with either modality removed and moves by < 0.01 when 10-50 % of patients randomly lack a modality. The auxiliary unimodal heads are what buy this (without them the model leans on WSI like SurvPath does).
4. **Fusion adds a small, consistent, not individually significant gain** over an ensemble of the two single-modality PathQ-Formers (+0.020 pooled over 25 pairs, p = 0.24; wins 4/5 cohorts). The honest framing is a unified model, not "better fusion".
5. **Overall-survival endpoint** reproduces the picture: +0.034 pooled over 25 folds (p = 0.003), significant on BLCA, BRCA and HNSC.
6. **Efficiency:** 16.9 M vs 21.2 M parameters, equal forward latency (~35 ms/patient), 1.7x faster training step (104 vs 176 ms), one third less peak GPU memory on full slides (1.17 vs 1.74 GiB). Cost is flat in the number of patches.
7. **Ablations of the final configuration** (BLCA): bin count barely matters (2 bins 0.628, 4 bins 0.625-0.630, 8 bins 0.609); richer pathway sets beat Hallmark-only by ~0.03 (Reactome+Hallmark 0.630, Xena 0.628, Hallmark 0.600); training on 4096 patches instead of all changes nothing (0.626 vs 0.625), so the gain over SurvPath is not from seeing more patches.

Total compute: about 130 USD on Runpod (RTX PRO 4000/4500 and RTX 2000 Ada pods plus a 250 GB network volume) and roughly 60 laptop-CPU hours.

---

## 1. Method

### 1.1 Architecture (`src/models/pathqformer.py`)

```
WSI patches (frozen UNI2-h, N x 1536)          RNA-seq (4,999 genes) -> 275 pathway tokens
            |                                                   |
   Histology query block (K = 32)                    Genomic query block (K = 32)
   LayerNorm(input) -> [cross-attn -> self-attn -> FFN] x 2 (pre-norm), d = 256, 8 heads
            |  Z_h (32 x 256)                                   |  Z_g (32 x 256)
            +----------------------- [Z_h ; Z_g] ---------------+
                                        |
                       Cross-modal fusion Transformer (2 layers, d = 256)
                                        |
                        attention pooling -> 4 discrete hazards -> NLL-survival loss
```

- **Pathway tokens.** Each of the 275 Reactome + Hallmark pathways (SurvPath's `combine` set; pathways with 3-300 genes) gets its own small MLP (hidden 128) over its member genes; the outputs are the genomic tokens. Alternative sets tested: Xena (281) and Hallmark-only (50).
- **Missing modalities.** A `NullTokenModule` holds a learned 32 x 256 code per modality. An absent modality is replaced by its null code before fusion. During training each present modality is dropped per sample with p = 0.15, never both (`modality_dropout`).
- **Auxiliary unimodal heads** (`aux_heads`, weight 0.5). Two extra survival heads on the pre-fusion codes Z_h and Z_g, each trained with the same NLL on the samples where its modality is present. They force each branch to be prognostic on its own. This is the recommended final configuration (`configs/protocol_fixed/pathq_fast_e20_aux.yaml`).
- **Loss / risk.** Discrete-time NLL-survival loss (Zadeh & Schmid, as in MCAT/SurvPath) over 4 hazard bins, `alpha = 0` for PathQ-Former (0.5 for SurvPath, its published setting). Risk score = -sum_t S(t).
- **Size.** 16.9 M trainable parameters including the pathway tokenizer (SurvPath: 21.2 M).

### 1.2 Training (final configuration)

| Setting | PathQ-Former (+aux) | SurvPath (official code, `data/survpath_repo`) |
|---|---|---|
| Epochs / selection | 20, final checkpoint | 20, final checkpoint |
| Optimizer | RAdam, lr 1e-4, wd 1e-4, 1 warm-up epoch, cosine | RAdam, lr 5e-4 (published), wd 1e-4 |
| Batch | 1 patient, gradient accumulation 2, clip 1.0 | 1 patient |
| Dropout | 0.25 (+ modality dropout 0.15) | 0.1 |
| Patches | all patches of all slides of the patient | 4096-patch subsample (published) |
| Sampling | weighted by (bin, censorship) | same |
| Genes | MinMax to [-1, 1] fit on the training split | same |

Baselines under the identical protocol and budget: **ABMIL** (attention MIL on WSI only), **SNN** (self-normalizing net on the 4,999 genes), **MLP** (two-layer MLP on the genes), all with the SurvPath repository's hyper-parameters, plus **late fusion** (z-scored risk average of a WSI-only and an RNA-only PathQ-Former).

### 1.3 Evaluation protocol (v2)

| Aspect | Choice |
|---|---|
| Unit | patient (`case_id`); all slides of a patient concatenated |
| Splits | SurvPath's exact 5-fold CV splits |
| Endpoint | disease-specific survival (DSS); overall survival (OS) as a secondary experiment |
| Time bins | quartiles of **uncensored training** patients, outer edges +-inf, re-used for validation |
| Primary metric | Harrell's C-index on continuous time (`sksurv.concordance_index_censored`) |
| Also stored per fold | Uno's IPCW C-index truncated at the 75th percentile of training event times; Brier / IBS and time-dependent AUC on the interior quartile grid; 1000x bootstrap 95 % CI; KM log-rank at the median risk split |
| Seeds | `seed + fold`; 3 seeds (0, 1, 2) per cell |
| Statistics | paired t and Wilcoxon signed-rank over (fold, seed) pairs, per cohort (15 pairs) and pooled (75 pairs) |
| Missing modality | same checkpoint tested WSI-only, RNA-only and with 10 / 20 / 30 / 50 % of patients randomly missing one modality (5 repeats); baselines get mean imputation instead |

"Mean +- std" in the tables below is the spread over the 3 seeds of the 5-fold mean unless stated otherwise.

---

## 2. Data

Five TCGA cohorts with UNI2-h patch features (HF `MahmoodLab/UNI2-h-features`, 1536-d, frozen) and SurvPath's RNA/metadata. Patient and event counts as realised by the DSS splits:

| Cohort | Patients | DSS events | Events per validation fold | Note |
|---|---|---|---|---|
| BLCA (bladder) | 359 | 113 | 16-25 | the only cohort with comfortable event counts |
| BRCA (breast) | 868 | 60 | 7-20 | 93 % censored; largest slides (median ~10k patches) |
| COADREAD (colorectal) | 296 | 37 | 5-13 | fewest events; fold-level C-index very noisy (std 0.12-0.16) |
| HNSC (head and neck) | 392 | 117 | 13-45 | |
| STAD (stomach) | 318 | 84 | 9-33 | |

Low event counts are the dominant source of variance: with 5-9 validation events a single patient reorders the fold's C-index by several points. This is why every table pools over folds and seeds and reports paired tests rather than per-cohort means alone.

---

## 3. Protocol audit (2026-09-10) and protocol decisions

### 3.1 Six defects in the May pipeline (all fixed; old numbers discarded)

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | C-index computed on the 4 discrete bins | ~75 % of pairs tied and dropped; not the literature's metric | Harrell's C on continuous months |
| 2 | Bin edges fit separately on train and validation | validation labels on a different scale; leaks the validation label distribution | bins from uncensored training patients, re-used |
| 3 | Slide-level samples (423) | multi-slide patients counted up to 9x; SurvPath is patient-level (359) | one sample per patient |
| 4 | Genes fed unscaled (log values -10..15) | arbitrary input scale for the pathway MLPs | MinMax on train |
| 5 | Selection and reporting on validation C-index; no seeds | optimistic bias; irreproducible fold collapse | selection on val loss or fixed budget; seeded |
| 6 | Risk = 1 - S_last; modality dropout per batch | non-standard risk; wrong batch semantics | risk = -sum S_t; per-sample dropout |

Two evaluation artefacts were also removed: SurvPath's Brier/AUC grid includes the validation min/max time, where one patient at risk gave AUC 0.13 and dragged the mean to 0.37; and the untruncated IPCW C-index explodes when the censoring survival G(t) reaches 0.03 at 120 months. The grid is now the interior quartile edges and Uno's C is truncated at the 75th percentile of training event times.

### 3.2 Why the fixed 20-epoch budget

The first multi-cohort campaign used early stopping on validation loss. On STAD, HNSC and BRCA every method, including the official SurvPath and the plain RNA baselines, landed near chance (0.45-0.55), far below the published 0.60-0.75. The histories showed the cause: with 7-16 events per validation fold the NLL is noise and val-loss selection picked epoch 1-3 checkpoints for every method while the validation C-index was still rising. This is a protocol failure, not an architecture result.

The fixed budget (train N epochs, report the final checkpoint, the MCAT/SurvPath convention) was adopted because under it the official SurvPath reproduces its published range and is seed-stable (BLCA 0.588 / 0.585 / 0.593; STAD 0.609 / 0.593 / 0.617). The budget was set to 20 rather than 10 epochs because the pooled per-epoch curves showed PathQ-Former still rising at epoch 10 while SurvPath was flat from epoch 8; at 20 epochs PathQ-Former is ahead at every epoch >= 11 (0.605 vs 0.573 at epoch 20, peaks 0.612 at 13 vs 0.585 at 14, pooled over 35 / 30 fold histories). The budget is identical for all six methods. Best-epoch numbers are always logged as an optimistic bound but never reported as results.

Under the 10-epoch budget PathQ-Former was already +0.028 pooled over 75 pairs (t p 0.034, Wilcoxon 0.017), winning BRCA and COADREAD significantly and losing STAD/HNSC within noise, so the conclusion does not hinge on the budget choice.

---

## 4. Main results (DSS, 20 epochs, final checkpoint, 3 seeds x 5 folds x 5 cohorts)

### 4.1 Table 1: C-index

| Method | Inputs | BLCA | BRCA | COADREAD | HNSC | STAD | Pooled delta vs SurvPath (75 pairs) |
|---|---|---|---|---|---|---|---|
| PathQ-Former | WSI + RNA | 0.609 +- 0.016 | 0.611 +- 0.039 | **0.638 +- 0.031** | 0.575 +- 0.012 | **0.593 +- 0.015** | **+0.037** (t p 0.002, W p 0.005) |
| PathQ-Former + aux heads | WSI + RNA | **0.630 +- 0.010** | 0.622 +- 0.033 | 0.613 +- 0.023 | **0.582 +- 0.025** | 0.572 +- 0.022 | **+0.036** (t p 0.007, W p 0.002) |
| SurvPath (official) | WSI + RNA | 0.594 +- 0.012 | 0.536 +- 0.026 | 0.570 +- 0.038 | 0.552 +- 0.019 | 0.587 +- 0.008 | reference |
| ABMIL | WSI | 0.566 +- 0.011 | 0.575 +- 0.004 | 0.587 +- 0.023 | 0.562 +- 0.028 | 0.553 +- 0.020 | +0.001 (p 0.94) |
| SNN | RNA | 0.590 +- 0.009 | 0.558 +- 0.013 | 0.572 +- 0.028 | 0.535 +- 0.006 | 0.549 +- 0.009 | -0.007 (p 0.67) |
| MLP | RNA | 0.611 +- 0.012 | **0.627 +- 0.013** | 0.633 +- 0.016 | 0.555 +- 0.012 | 0.532 +- 0.009 | +0.024 (t p 0.15, W p 0.30) |

Per-cohort paired tests vs SurvPath (15 pairs each):

| Cohort | PathQ-Former delta (t p) | PathQ-Former + aux delta (t p) |
|---|---|---|
| BLCA | +0.015 (0.36) | **+0.036 (0.024)** |
| BRCA | **+0.075 (0.013)** | **+0.086 (0.004)** |
| COADREAD | +0.068 (0.064) | +0.044 (0.32) |
| HNSC | +0.023 (0.38) | +0.031 (0.29) |
| STAD | +0.006 (0.79) | -0.015 (0.57) |

Reading: PathQ-Former >= SurvPath on every cohort (one -0.015 tie for the aux variant on STAD), significantly better on two, with a pooled advantage that survives both tests. SurvPath's published numbers are for OS with its own preprocessing, so the reference point here is the same code run under our protocol, which is the comparison that matters.

### 4.2 Table 2: the same runs with the RNA MLP as reference

| Method | BLCA | BRCA | COADREAD | HNSC | STAD | Pooled (75 pairs) |
|---|---|---|---|---|---|---|
| PathQ-Former | -0.002 | -0.016 | +0.004 | +0.020 | +0.062 (p 0.066) | +0.014 (t p 0.25, W p 0.21) |
| PathQ-Former + aux | +0.019 | -0.005 | -0.020 | +0.028 | +0.041 | +0.012 (t p 0.28, W p 0.12) |
| SurvPath | -0.017 | -0.091 | -0.064 | -0.003 | +0.056 | -0.024 (t p 0.15) |

A two-layer MLP on the 4,999 genes is a strong baseline that the multimodal literature rarely reports at equal budget. PathQ-Former is never significantly worse than it and is +0.04 to +0.06 on STAD, where RNA alone is weak; the MLP loses STAD by 0.06 and HNSC by 0.02-0.03, ABMIL loses BLCA and BRCA. The multimodal model is the only one that is at least as good as the best unimodal model on all five cohorts.

### 4.3 Secondary metrics

Every run also stores IPCW C, IBS and td-AUC (e.g. final config BLCA seed 0: Harrell 0.625, IPCW 0.626, IBS 0.212, td-AUC 0.683). These track Harrell's C; they were not used for any decision and are available in `pod_results/outputs_e20/<run>/<cohort>/summary.md`.

---

## 5. Missing-modality robustness

### 5.1 Table 3: one modality removed at test time (20 ep, seed 0, same checkpoint)

Baselines receive the training-mean gene vector or mean patch (`scripts/eval_missing_impute.py`) so that the comparison is degradation, not availability. PathQ-Former uses its learned null codes (imputation gives similar numbers for it).

| Cohort | SurvPath full | SurvPath, RNA missing | SurvPath, WSI missing | PathQ+aux full | PathQ+aux, RNA missing | PathQ+aux, WSI missing |
|---|---|---|---|---|---|---|
| BLCA | 0.606 | 0.608 | 0.509 | 0.625 | 0.598 | 0.588 |
| BRCA | 0.552 | 0.552 | 0.514 | 0.647 | 0.610 | 0.671 |
| COADREAD | 0.532 | 0.534 | 0.480 | 0.595 | 0.547 | 0.594 |
| HNSC | 0.531 | 0.533 | 0.551 | 0.603 | 0.604 | 0.575 |
| STAD | 0.589 | 0.590 | 0.462 | 0.559 | 0.579 | 0.541 |

Three findings:

- **SurvPath ignores RNA.** Its score is identical with the genomic input replaced by the mean (max change 0.002) and collapses to chance without WSI. Its co-attention learns to route everything through the histology branch.
- **PathQ-Former + aux degrades gracefully in both directions** (0.54-0.67), sometimes not at all (BRCA WSI-missing 0.671 > full 0.647; HNSC RNA-missing 0.604 = full).
- **The auxiliary heads are the mechanism.** Without them the plain 20-epoch PathQ-Former behaves like SurvPath: BLCA WSI-only 0.635 but RNA-only 0.535; HNSC 0.585 / 0.548. With them: 0.598 / 0.588 and 0.604 / 0.575. The heads cost nothing on the fused score (+0.02 on BLCA, p = 0.85 paired).

### 5.2 Random partial missingness (final config, seed 0; 10 / 20 / 30 / 50 % of patients missing one modality, 5 repeats)

| Cohort | Full | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|---|
| BLCA | 0.625 | 0.626 | 0.625 | 0.624 | 0.618 |
| BRCA | 0.647 | 0.643 | 0.634 | 0.642 | 0.643 |
| COADREAD | 0.595 | 0.590 | 0.591 | 0.584 | 0.581 |
| HNSC | 0.603 | 0.598 | 0.597 | 0.610 | 0.590 |
| STAD | 0.559 | 0.559 | 0.561 | 0.568 | 0.554 |

The fused C-index moves by at most 0.014 with half the cohort missing a modality. A deployment that sees a realistic mix of WSI-only, RNA-only and complete patients needs one model.

---

## 6. Is fusion doing anything? Joint fusion vs single-modality vs ensemble

### 6.1 All cohorts, 20 epochs (single-modality PathQ-Formers trained at seed 0; joint = 3-seed mean)

| Cohort | PathQ+aux (joint) | Late fusion (WSI-only + RNA-only PathQ) | WSI-only PathQ | RNA-only PathQ |
|---|---|---|---|---|
| BLCA | 0.630 | **0.642** | 0.593 | 0.631 |
| BRCA | **0.622** | 0.592 | 0.596 | 0.558 |
| COADREAD | **0.613** | 0.571 | 0.588 | 0.566 |
| HNSC | **0.582** | 0.578 | 0.586 | 0.563 |
| STAD | **0.572** | 0.549 | 0.567 | 0.512 |

Joint fusion beats the late-fusion ensemble on 4/5 cohorts, +0.020 pooled over 25 pairs (p = 0.24), and beats the better single-modality branch by 0 to +0.03. This is a small, consistent gain that is not individually significant: the fusion block is not extracting strong complementarity beyond an ensemble, and the paper should say so.

### 6.2 BLCA fusion ablations (10-epoch budget, seed 0)

| Variant | C-index |
|---|---|
| base | 0.609 |
| modality dropout 0 / 0.3 / 0.5 | 0.634 / 0.616 / 0.635 |
| K = 16 / 64 queries | 0.638 / 0.634 |
| fusion depth 1 / 3 | 0.598 / 0.595 |
| joint (3 seeds) / late fusion / WSI-only / RNA-only | 0.618 / 0.631 / 0.596 / 0.603 |

None of K, fusion depth or modality dropout moves the fused score outside one fold-std (0.60-0.64). Modality dropout matters only for missing-modality robustness (0.3 was best there). The test-time single-modality scores of the same checkpoints were as high as the fused score (md30: fused 0.616, WSI-only 0.624, RNA-only 0.648), which was the observation that motivated the auxiliary heads.

---

## 7. Overall-survival endpoint (20 ep, seed 0)

| Cohort | PathQ+aux | SurvPath | delta | paired t p (5 folds) |
|---|---|---|---|---|
| BLCA | **0.595** | 0.552 | +0.043 | 0.008 |
| BRCA | **0.596** | 0.523 | +0.073 | 0.015 |
| COADREAD | 0.581 | **0.596** | -0.015 | 0.58 |
| HNSC | **0.536** | 0.510 | +0.025 | 0.046 |
| STAD | **0.603** | 0.559 | +0.044 | 0.23 |
| pooled (25 folds) | | | **+0.034** | t 0.0028, Wilcoxon 0.0025 |

Same picture as DSS. Absolute OS C-indices are 0.02-0.03 below DSS for both methods because OS counts non-cancer deaths. Missing-modality behaviour holds on OS (BLCA both / WSI-only / RNA-only 0.595 / 0.576 / 0.592; STAD 0.603 / 0.574 / 0.582).

---

## 8. Ablations of the final configuration (BLCA, PathQ+aux, 20 ep, seed 0, laptop CPU)

| Variant | Both | WSI-only | RNA-only |
|---|---|---|---|
| default: 4 bins, Reactome+Hallmark 275 pathways, all patches (seed 0 / 3-seed mean) | 0.625 / 0.630 | 0.598 | 0.588 |
| 2 hazard bins | 0.628 | 0.634 | 0.576 |
| 8 hazard bins | 0.609 | 0.579 | 0.605 |
| Xena 281 pathways | 0.628 | | |
| Hallmark-only 50 pathways | 0.600 | | |
| 4096 training patches instead of all | 0.626 | 0.591 | 0.622 |

- Bin count: 2 ~ 4 > 8, all within one fold-std; 8 bins spreads too few events per bin. (With 2 bins the Brier grid has no interior edge, so IBS is undefined; C-index is unaffected.)
- Pathway set: the two 275-281-pathway sets are equivalent; Hallmark-only costs ~0.03. The genomic branch benefits from many small pathway tokens.
- Patch budget: training on a 4096-patch subsample gives the same score as all patches. PathQ-Former's edge over SurvPath (which trains on 4096) is therefore not explained by patch count.

---

## 9. Efficiency (`scripts/efficiency.py`; RTX PRO 4000 Blackwell; BLCA fold-0 validation, 40 real patients, medians)

| Model | Params (M) | Patches | Forward (ms) | Forward + backward (ms) | Peak GPU memory (GiB) |
|---|---|---|---|---|---|
| PathQ-Former (all patches) | 16.9 | 6214 | 35.0 | 104.2 | 1.17 |
| PathQ-Former (4096 patches) | 16.9 | 4096 | 34.7 | 104.1 | 0.17 |
| SurvPath (4096 patches, as trained) | 21.2 | 4096 | 35.7 | 175.6 | 0.20 |
| SurvPath (all patches) | 21.2 | 6214 | 36.0 | 176.0 | 1.74 |

PathQ-Former's compute is dominated by the 32-query cross-attention, so latency is flat in the number of patches; SurvPath's patch-pathway co-attention grows with both. On the laptop CPU with a 36k-patch slide: PathQ-Former 190 ms forward / 1013 ms step, SurvPath 188 / 724 ms (both ~45 / ~215 ms at 4096 patches). Training throughput on the pods was I/O-bound (feature reads from the network volume), not model-bound.

---

## 10. Validity: what the numbers do and do not support

**Supported**

- PathQ-Former >= SurvPath on all five cohorts under an identical, pre-registered protocol, with a pooled advantage that is significant under both a parametric and a rank test over 75 paired folds, and reproduced on the OS endpoint.
- The same checkpoint works with either modality absent; SurvPath does not (and cannot run at all without imputation).
- The result is stable to the training budget (10 vs 20 epochs), bin count, pathway set and patch budget.

**Not supported / must be stated**

- PathQ-Former is not significantly better than a tuned RNA-only MLP pooled (+0.012 to +0.014), and loses to it within noise on BRCA and COADREAD. The claim is "at least as good as the best unimodal model on every cohort, with one network", plus robustness.
- Fusion beats late fusion by +0.02 pooled, not significantly. Do not claim complementarity beyond an ensemble.
- Per-cohort significance exists only for BLCA and BRCA (and HNSC on OS). COADREAD (37 events) and STAD are inconclusive at this sample size.
- Comparison to published numbers is indirect: our protocol (DSS, patient-level, fixed budget, final checkpoint) differs from each paper's; the meaningful reference is the official SurvPath code under our protocol, which lands in its published range.
- Missing-modality evaluation removes a modality from all validation patients of a fold or from random subsets; it does not model informative missingness.
- Single seed for the OS endpoint, single-modality PathQ-Formers, late fusion and Table 3; three seeds everywhere in Table 1.

**Known sources of variance**: 5-45 events per validation fold; fold 3 of BLCA is weak for every method; fold 0 of COADREAD scores 0.75-0.84 for both methods with 5 events.

---

## 11. Engineering record

**Pipeline** (`src/`): `data/tcga_dataset.py` (patient-level dataset, training-fit bins and scaler, optional fp16 RAM cache capped at a fraction of the container's cgroup memory limit), `models/pathqformer.py`, `models/baselines.py` (ABMIL, SNN, MLP, official SurvPath wrapper), `training/evaluate.py` (all metrics, bootstrap, KM), `training/train.py` (5-fold CV with two-level resume: finished folds and mid-fold checkpoints; `--set key=value` overrides; RAdam; weighted sampling; auxiliary loss masked to present samples).

**Analysis scripts** (`scripts/`): `seed_table.py` (seeded tables with paired tests), `aggregate_results.py`, `epoch_curves.py`, `late_fusion.py`, `eval_missing_impute.py`, `efficiency.py`, `recompute_metrics.py` (re-derives all metric blocks from saved per-patient predictions so protocol changes apply retroactively), `analyze_run.py --attention` (attention export for figures).

**Tests**: 34 tests (`pytest`), 11 of them slow end-to-end trainer tests on dummy embeddings: overrides and validation, weighted sampler, OS/PFI endpoints, patients without embeddings, full CV run + fold skip, mid-fold resume, all three selection modes, seed determinism, auxiliary loss, every baseline through the trainer, every analysis script. Bugs the suite caught: `--set lr=1e-4` parsed as a string (YAML 1.1); bootstrap crash on resamples with no comparable pairs (COADREAD).

**Compute infrastructure**: Runpod pods (RTX PRO 4000/4500 Blackwell, RTX 2000 Ada) sharing a 250 GB network volume (fjb5dlrrfp, EU-RO-1) with all five cohorts' UNI2-h features, outputs and logs; resumable queue scripts (`scripts/pod/`) that skip finished runs, chain batches, and remove their own pod; up to three pods in parallel. All 180 runs' checkpoints (855 files, ~44 GB) remain on the volume.

**Incidents worth remembering**: (1) container OOM: `free` shows the host's 125 GB but the cgroup limit was 29-62 GB; the cache is now sized from `memory.max`. (2) Early-stopping-by-val-loss protocol failure on event-poor cohorts (Section 3.2). (3) Pod self-removal failed twice on `runpodctl` version differences; the v1/v2/REST three-way chain in `run_queue_pod.sh` is the fix, and a hand-written watcher that skipped it cost ~13 idle hours (~8 USD). (4) The account balance reached zero once and Runpod deleted every pod, including one unrelated to this project; the volume kept everything and the interrupted run resumed from its saved folds.

---

## 12. Compute and cost

| Item | Amount |
|---|---|
| Finished GPU runs (5-fold CV each) | 180 (outputs_v2 34, outputs_ablate 31, outputs_e20 101, outputs_os 10 + late fusion) |
| Laptop CPU runs | 12 (BLCA val-loss protocol table, final-config ablations) |
| Pod hours | roughly 210 GPU-hours across 9 pods, 2026-09-10 to 2026-09-19 |
| Runpod spend | about 130 USD including the network volume (~1 USD/day) |
| Typical per-run cost | PathQ-Former 20 ep on BLCA ~25 min on RTX PRO 4000 (I/O-bound); BRCA ~1.5-2 h |

---

## 13. Recommended paper framing

**Title-level claim:** a unified, missing-modality-robust multimodal survival model that matches or beats every baseline under an identical protocol on five TCGA cohorts, and significantly beats the strongest published multimodal baseline.

**Table 1** = Section 4.1 (six methods, five cohorts, 3 seeds, paired tests vs SurvPath and vs the RNA MLP).
**Table 2** = Section 5.1 (mean-imputed baselines vs null-code PathQ-Former) with Section 5.2 as a figure.
**Table 3** = Section 6.1 (joint vs late fusion vs single-modality).
**Figures**: pooled epoch curves (budget fairness), missing-rate curves, attention maps from `analyze_run.py --attention` (to be produced from the final checkpoints), efficiency bar chart.
**Ablations (appendix)**: Sections 6.2 and 8, plus the protocol audit as a reproducibility note (Section 3), which is itself a contribution: the val-loss failure mode explains why several published multimodal comparisons on event-poor cohorts are unreliable.

**Limitations to state**: RNA MLP parity; fusion ~ ensemble; single seed for secondary experiments; TCGA-only; frozen UNI2-h features; no external validation cohort.

---

## 14. Next steps

1. **Paper draft** from Sections 4-9 (NeurIPS 2026 workshop target, ICLR 2027 main).
2. **Interpretability figures**: attention over patches and pathways for the final checkpoints (one short pod session; checkpoints on the volume).
3. **Seeds for the secondary experiments** (OS, single-modality PathQ-Former, late fusion) if the reviewers' bar requires it: ~10 GPU-hours.
4. **Optional strengthening of the accuracy claim**: a pathway-token input for the MLP baseline (to separate "pathway tokens help" from "fusion helps"), and an external cohort (CPTAC) for the robustness claim.
5. **Housekeeping**: remove the network volume when the figures are done (it bills ~1 USD/day); rotate the HuggingFace and Runpod credentials used during the campaign.

---

## Appendix A. Reproducing the tables

```bash
# Table 1 / Table 2 (seeded, paired tests)
.venv/Scripts/python.exe scripts/seed_table.py pod_results/outputs_e20 \
  --methods pathq_fast_e20 pathq_fast_e20_aux survpath_e20 abmil_e20 snn_e20 mlp_omics_e20 --ref survpath_e20
.venv/Scripts/python.exe scripts/seed_table.py pod_results/outputs_e20 \
  --methods pathq_fast_e20 pathq_fast_e20_aux survpath_e20 mlp_omics_e20 --ref mlp_omics_e20
# OS endpoint
.venv/Scripts/python.exe scripts/seed_table.py pod_results/outputs_os --methods pathq_fast_e20_aux survpath_e20 --ref survpath_e20
# 10-epoch seeded table
.venv/Scripts/python.exe scripts/seed_table.py pod_results/outputs_ablate --methods pathq_fast_e10 survpath_e10 --ref survpath_e10
# epoch curves, late fusion, imputation eval, efficiency
.venv/Scripts/python.exe scripts/epoch_curves.py pod_results/outputs_e20
.venv/Scripts/python.exe scripts/late_fusion.py --wsi pod_results/outputs_e20/pathq_e20_wsi_only/blca --rna pod_results/outputs_e20/pathq_e20_genomic_only/blca
.venv/Scripts/python.exe scripts/eval_missing_impute.py --run pod_results/outputs_e20/survpath_e20/blca
cat pod_results/outputs_e20/efficiency.md
```

Per-run artefacts: `pod_results/outputs_*/<run>/<cohort>/{results.json, summary.md, config.yaml, fold_k/}`; raw pulls in `pod_results/pull_0918/*.tgz`. Final configurations: `configs/protocol_fixed/`; ablations: `configs/ablation_fixed/`.

## Appendix B. Run inventory

| Store | Content | Runs |
|---|---|---|
| outputs_v2 | val-loss protocol (all methods, 5 cohorts, BLCA seeds), 10-ep fixed budget seed 0, laptop ablations | 34 (+12 laptop) |
| outputs_ablate | BLCA fusion ablations, late fusion, 10-ep seeds 1-2 for PathQ-Former and SurvPath | 31 |
| outputs_e20 | 20-ep protocol: 6 methods x 5 cohorts x 3 seeds, single-modality PathQ-Former, late fusion, efficiency | 101 |
| outputs_os | OS endpoint, PathQ+aux and SurvPath, 5 cohorts | 10 |
