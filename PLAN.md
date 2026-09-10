# PathQ-Former — Master Plan

**Project**: Q-Former-based multimodal fusion for WSI + genomics cancer survival prediction
**Target venues**: NeurIPS 2026 Workshop (Oct deadline) → ICLR 2027 (Sep deadline)
**Status**: Phase 2b — re-running BLCA under the corrected v2 protocol (see EXPERIMENT_LOG.md 2026-09-10); Phases 4E and 5 evaluation are now automatic in every run
**Last updated**: 2026-09-10


> **2026-09-10 protocol note.** The May pipeline had six defects (C-index on discrete bins,
> val-fit bins, slide-level samples, unscaled genes, val-C-index selection, non-standard risk).
> All BLCA numbers below are superseded. The v2 pipeline (`configs/*_v2.yaml`) follows the
> MCAT/SurvPath protocol exactly and evaluates missing-modality robustness, IPCW C-index,
> IBS, td-AUC, bootstrap CIs and KM log-rank for every run. Colab is no longer required:
> a BLCA 5-fold run takes about 2 h on the local 12-core CPU.

---

## Architecture Overview

```
WSI Patches (frozen UNI2-h)         Genomic Pathways (learnable)
      [N x 1536]                        [4999 genes → 275 pathways]
          │                                       │
          ▼                                       ▼
   Input Projection                     Pathway Tokenizer (MLP per pathway)
      [N x 256]                              [275 x 256]
          │                                       │
          ▼                                       ▼
┌──────────────────────┐            ┌──────────────────────┐
│  Histology Query     │            │  Genomic Query       │
│  Block (K=32)        │            │  Block (K=32)        │
│  L=2 layers of:      │            │  L=2 layers of:      │
│  cross-attn→self-attn│            │  cross-attn→self-attn│
│  →FFN (pre-norm)     │            │  →FFN (pre-norm)     │
└──────────────────────┘            └──────────────────────┘
          │ Z_h [32 x 256]                    │ Z_g [32 x 256]
          └──────────┬────────────────────────┘
                     ▼
              [Z_h ; Z_g] → [64 x 256]
                     │
          ┌──────────▼──────────┐
          │  Cross-Modal Fusion │
          │  Transformer (M=2)  │
          │  (self-attention)   │
          └──────────┬──────────┘
                     │ [64 x 256]
                     ▼
             Attention Pooling
                     │ [256]
                     ▼
            Hazard Classifier
                     │
                     ▼
           Hazard Logits [4 bins]
                → NLL-Survival Loss
```

**Trainable parameters**: ~16.7M (10.4M pathway tokenizer + 6.3M PathQ-Former)
**WSI encoder**: UNI2-h (ViT-H/14, 1536-d, frozen pre-extracted embeddings from MahmoodLab/UNI2-h-features)
**Missing-modality**: Learned null token embeddings + modality dropout (p=0.15) during training

---

## Phase 1: BLCA Baseline — COMPLETE

**Goal**: Establish baseline performance on BLCA with default hyperparameters
**Date**: 2026-05-19

| Parameter | Value |
|-----------|-------|
| LR | 2e-4 |
| Weight decay | 1e-5 |
| Dropout | 0.1 |
| Modality dropout | 0.15 |
| Batch size | 1 |
| Grad accumulation | 1 |
| Warmup | None |
| Patience | 5 |

**Result**: Mean C-Index = **0.6052 +/- 0.0200**
- Per-fold: [0.5756, 0.6340, 0.5931, 0.6062, 0.6173]
- Overfitting: peaks epoch 2-3, then declines rapidly
- Competitive with MCAT (~0.58-0.61), below SurvPath/MMP

---

## Phase 2: BLCA Hyperparameter Tuning — IN PROGRESS

**Goal**: Fix overfitting, improve C-Index above SurvPath (~0.60-0.63)
**Date**: 2026-05-19

| Parameter | Baseline | Tuned | Rationale |
|-----------|----------|-------|-----------|
| LR | 2e-4 | **5e-5** | Reduce overshooting with batch_size=1 |
| Weight decay | 1e-5 | **1e-4** | Stronger L2 regularization |
| Dropout | 0.1 | **0.25** | Reduce overfitting in query/fusion blocks |
| Grad accumulation | 1 | **8** | Effective batch_size=8 for smoother gradients |
| Warmup | None | **2 epochs** | Linear warmup prevents early instability |
| Patience | 5 | **3** | Model improves gradually now, stop earlier |

**Results so far**:
| Fold | Baseline | Tuned | Diff |
|------|----------|-------|------|
| 0 | 0.5756 | **0.7068** | +0.131 |
| 1 | 0.6340 | **0.6606** | +0.027 |
| 2 | 0.5931 | **0.6083** | +0.015 |
| 3 | 0.6062 | **0.3977** | -0.209 |
| 4 | 0.6173 | **0.7025** | +0.085 |
| **Mean** | **0.6052 ± 0.020** | **0.6152 ± 0.114** | **+0.010** |

### Verdict
Mean improved only slightly (+0.01) because fold 3 collapsed to 0.3977. But excluding the outlier, the remaining 4 folds averaged **0.6696** — a strong +0.064 gain over baseline. The problem is clear:
- Warmup (2 epochs) + patience (3) = fold 3 only got 3 post-warmup epochs before being killed
- The tuned config is too conservative for hard splits that need more time

### Phase 2B: Hybrid Config — NEXT
Fix the fold-3 problem without losing the gains on other folds:
| Parameter | Tuned | Hybrid | Rationale |
|-----------|-------|--------|-----------|
| Warmup | 2 epochs | **1 epoch** | Less warmup = more real training before patience kicks in |
| Patience | 3 | **5** | Give slow folds more room to recover |
| LR | 5e-5 | 5e-5 | Keep (was too high at 2e-4) |
| Dropout | 0.25 | 0.25 | Keep (reduces overfitting) |
| Grad accum | 8 | 8 | Keep (smooth gradients) |
| Weight decay | 1e-4 | 1e-4 | Keep |

- [ ] Create hybrid config YAML
- [ ] Create Colab notebook for hybrid run
- [ ] Run hybrid 5-fold CV on BLCA
- [ ] If mean > 0.65 with std < 0.05, move to Phase 3

---

## Phase 3: Multi-Cancer Evaluation — PLANNED

**Goal**: Validate generalization across 5 TCGA cancer types using tuned config
**Timeline**: After Phase 2 completes

### Cancer types and embedding sizes
| Cancer | TCGA Code | Slides | Embedding Size (approx) |
|--------|-----------|--------|-------------------------|
| Bladder | BLCA | 423 | ~30 GB |
| Breast | BRCA | ~1,100 | ~65 GB |
| Stomach | STAD | ~450 | ~18 GB |
| Colorectal | COADREAD | ~550 | ~22 GB |
| Head & Neck | HNSC | ~500 | ~17 GB |

### Steps
- [ ] Download BRCA embeddings from HuggingFace (largest, do first to test disk limits)
- [ ] Download STAD, COADREAD, HNSC embeddings
- [ ] Create per-cancer config YAMLs (same tuned hyperparams, different cancer_type)
- [ ] Create Colab notebook template for multi-cancer runs
- [ ] Run 5-fold CV on each cancer type
- [ ] Compile cross-cancer results table

### Expected output
| Cancer | PathQ-Former | MCAT | SurvPath | MMP |
|--------|-------------|------|----------|-----|
| BLCA | ? | ~0.58-0.61 | ~0.60-0.63 | ~0.61-0.64 |
| BRCA | ? | ~0.60-0.63 | ~0.62-0.65 | ~0.63-0.66 |
| STAD | ? | ~0.55-0.58 | ~0.57-0.60 | ~0.58-0.61 |
| COADREAD | ? | ~0.56-0.59 | ~0.58-0.61 | ~0.59-0.62 |
| HNSC | ? | ~0.58-0.61 | ~0.60-0.63 | ~0.61-0.64 |

---

## Phase 4: Ablation Studies — PLANNED

**Goal**: Understand contribution of each architectural choice. Run on BLCA (fastest cancer type).

### 4A. Number of Queries (K)
Vary K in {16, 32, 64, 128}. Measures how much information the query bottleneck retains.
- [ ] K=16: fewer queries, stronger compression
- [ ] K=32: current default
- [ ] K=64: more queries, richer but more expensive
- [ ] K=128: upper bound

### 4B. Modality Dropout Rate
Vary modality_dropout in {0.0, 0.05, 0.15, 0.25}.
- Tests whether modality dropout acts as regularization or just noise.
- [ ] p=0.0 (no modality dropout)
- [ ] p=0.05
- [ ] p=0.15 (current default)
- [ ] p=0.25

### 4C. Architecture Depth
- [ ] Query layers: L in {1, 2, 3}
- [ ] Fusion layers: M in {1, 2, 3}
- Keep total params roughly constant by adjusting hidden_dim if needed

### 4D. Fusion Strategy
- [ ] No fusion — concatenate Z_h, Z_g and pool directly (lower bound)
- [ ] Cross-attention fusion — queries attend to other modality's queries (Q from one, KV from other)
- [ ] Current self-attention fusion — default
- [ ] Bilinear fusion — outer product interaction (SurvPath style, for comparison)

### 4E. Component Ablation
- [ ] WSI-only (no genomics, null token always)
- [ ] Genomics-only (no WSI, null token always)
- [ ] Full model (both modalities)
- Quantifies the contribution of each modality

### 4F. Number of Survival Bins
Vary num_bins in {2, 4, 8}. Default is 4 (following MCAT/SurvPath).
- [ ] 2 bins — coarsest, binary-like
- [ ] 4 bins — current default
- [ ] 8 bins — finer temporal resolution, but harder to learn with small data

### 4G. Pathway Composition Source
- [ ] combine (Reactome + Hallmarks, 275 pathways) — current default
- [ ] hallmarks only (~50 pathways) — fewer, broader pathways
- [ ] xena — alternative grouping
- Tests whether pathway granularity matters

### Automation consideration
Consider Karpathy's autoresearch framework to automate sweeps. Good fit for:
- Fixed evaluation protocol (5-fold CV, C-Index)
- Discrete hyperparameter grid
- Each run is independent
Defer until Phase 3 is done and we have a stable pipeline.

---

## Phase 5: Missing-Modality Experiments — PLANNED

**Goal**: Demonstrate robustness to missing modalities at test time (unique selling point vs MCAT/SurvPath)

### Experimental protocol
Train ONE model with modality_dropout=0.15 (already done in tuned config). At test time:
1. **Both modalities** — standard evaluation
2. **WSI only** — pass genomic_features=None (model uses learned null token)
3. **Genomics only** — pass wsi_features=None (model uses learned null token)

### Expected results table (Table 3 in paper)
| Test Condition | PathQ-Former | MCAT* | SurvPath* |
|---------------|-------------|-------|----------|
| Both | 0.XX | 0.XX | 0.XX |
| WSI only | 0.XX | N/A** | N/A** |
| Genomics only | 0.XX | N/A** | N/A** |

*Baselines require BOTH modalities — they crash or degrade without retraining.
**This is our key advantage — graceful degradation without retraining.

### Steps
- [ ] Implement test-time modality masking in evaluation code
- [ ] Run missing-modality eval on all 5 cancer types
- [ ] Compare against unimodal baselines (ABMIL for WSI-only, SNN for genomics-only)
- [ ] Plot graceful degradation curve (% of patients missing a modality vs C-Index)
- [ ] Simulate real-world scenario: randomly mask 10-50% of test patients' modalities

---

## Phase 5B: Statistical Rigor & Additional Metrics — PLANNED

**Goal**: Go beyond C-Index to make the evaluation bulletproof for top venues

### Additional evaluation metrics
- [ ] **Time-dependent AUROC** (td-AUC): Measures discrimination at specific time horizons (e.g., 1yr, 3yr, 5yr). More informative than global C-Index.
- [ ] **Integrated Brier Score (IBS)**: Measures calibration — are predicted survival probabilities accurate, not just ranked correctly?
- [ ] **Kaplan-Meier stratification**: Split patients into risk groups (high/medium/low by predicted risk), plot KM curves, report log-rank p-value. Visual proof the model separates survivors from non-survivors.

### Statistical significance
- [ ] **Paired t-test or Wilcoxon signed-rank** across 5 folds: Is PathQ-Former vs SurvPath improvement statistically significant (p < 0.05)?
- [ ] **95% confidence intervals** on mean C-Index (not just mean +/- std)
- [ ] **Bootstrap resampling** (1000 iterations) within each fold for per-fold CIs
- [ ] Report p-values in all comparison tables

### Per-subtype analysis
- [ ] Break down results by cancer subtype using oncotree_code from metadata
- [ ] Identify which subtypes benefit most from multimodal fusion
- [ ] Stratify by clinical covariates: age, sex, tumor stage (if available)

### Interpretability & visualization
- [ ] **Query attention heatmaps**: Which WSI patches do queries attend to most? Overlay on tissue thumbnail.
- [ ] **Pathway importance ranking**: Which genomic pathways contribute most to survival prediction? (via attention weights or gradient-based attribution)
- [ ] **Cross-modal attention patterns**: In the fusion block, do histology queries preferentially attend to specific pathway queries? Visualize the 32x32 cross-attention matrix.
- [ ] **t-SNE/UMAP of fused representations**: Do risk groups form separable clusters?

---

## Phase 6: Baseline Reproduction — PLANNED

**Goal**: Fair comparison using identical splits, embeddings, and evaluation protocol

### Baselines to reproduce
| Method | Priority | Notes |
|--------|----------|-------|
| MCAT | High | SurvPath repo has implementation |
| SurvPath | High | Official repo, our splits come from here |
| ABMIL | Medium | WSI-only baseline, simple to run |
| MMP | Low | Code not yet public (ICML 2024) |

### Steps
- [ ] Run SurvPath's train scripts with their default config on our 5 cancer types
- [ ] Run MCAT using SurvPath's provided implementation
- [ ] Verify our numbers match published numbers (+/- 0.01-0.02)
- [ ] Use same UNI2-h embeddings for all methods (fair comparison)

### Note on embeddings
Published SurvPath/MCAT results use ResNet-50 or CONCH features. We use UNI2-h (stronger encoder). Running baselines with UNI2-h shows our architectural advantage, not just encoder advantage. Running baselines with their original encoder shows the combined improvement.

### Survival endpoints
The metadata supports multiple survival endpoints. We should report on at least two:
- [ ] **DSS** (Disease-Specific Survival) — current primary endpoint
- [ ] **OS** (Overall Survival) — most commonly reported in clinical literature
- [ ] **PFI** (Progression-Free Interval) — optional, if time permits

---

## Phase 7: Paper Writing — PLANNED

**Target**: NeurIPS 2026 Workshop (Oct deadline)

### Paper structure
1. **Introduction**: Clinical motivation (missing modalities in practice), Q-Former for pathology
2. **Related Work**: MCAT, MOTCat, SurvPath, MMP, BLIP-2/Q-Former lineage
3. **Method**: PathQ-Former architecture, pathway tokenization, modality dropout, NLL loss
4. **Experiments**:
   - Table 1: 5-cancer comparison vs baselines (Phase 3+6)
   - Table 2: Ablation studies (Phase 4)
   - Table 3: Missing-modality robustness (Phase 5)
   - Figure: Architecture diagram, training curves, attention visualization
5. **Discussion**: When PathQ-Former helps, limitations, clinical applicability

### Figures needed
- [ ] Architecture diagram (main figure, TikZ or draw.io)
- [ ] Training curves: baseline vs tuned (show reduced overfitting)
- [ ] Cross-cancer bar chart (C-Index comparison, 5 cancers x N methods)
- [ ] Missing-modality degradation curve (% missing vs C-Index)
- [ ] Attention heatmap on WSI (which patches get high query attention)
- [ ] Kaplan-Meier curves (high vs low risk groups, with p-value)
- [ ] Ablation bar chart (K, depth, fusion strategy)
- [ ] Cross-modal attention matrix visualization (histology ↔ pathway queries)

### Tables needed
- Table 1: Multi-cancer comparison (5 cancers x methods, C-Index + CI + p-values)
- Table 2: Ablation study (K, dropout, depth, fusion, bins, pathway source)
- Table 3: Missing-modality robustness (both/WSI-only/genomics-only)
- Table 4: Multiple survival endpoints (DSS vs OS)
- Table S1 (supplement): Per-fold results for all experiments
- Table S2 (supplement): Per-subtype breakdown

---

## Phase 8: Advanced Extensions — STRETCH GOALS

**Goal**: Strengthen the paper with novel contributions beyond what exists in the literature. These are stretch goals — pursue after core experiments (Phases 1-7) are solid.

### 8A. External Validation (CPTAC)
TCGA-only results are standard but reviewers increasingly ask for external validation.
- [ ] Download CPTAC cohorts (lung, breast, colon — overlap with TCGA cancer types)
- [ ] Extract UNI2-h embeddings from CPTAC WSIs (requires running UNI2-h inference, not just downloading)
- [ ] Train on TCGA, test on CPTAC — zero-shot generalization
- **Impact**: Very strong if it works. Shows model isn't overfit to TCGA artifacts.
- **Risk**: CPTAC has different staining, scanners, protocols. May need domain adaptation.

### 8B. Learnable Query Initialization (vs Random)
Current queries are initialized randomly (N(0, 0.02)). Alternatives:
- [ ] Initialize from WSI prototype embeddings (mean of training patches)
- [ ] Initialize from pathway embeddings (mean of pathway tokenizer outputs)
- [ ] Use K-Means centroids from training data as initialization
- **Hypothesis**: Informed initialization may converge faster and improve on small datasets

### 8C. Multi-Task Learning
Add auxiliary prediction tasks alongside survival:
- [ ] Cancer subtype classification (oncotree_code) — free labels from metadata
- [ ] Tumor grade prediction (if available in clinical data)
- [ ] Molecular subtype prediction (e.g., PAM50 for BRCA)
- **Hypothesis**: Auxiliary tasks regularize the fusion representation and improve generalization

### 8D. Feature-Space Data Augmentation
Current transforms.py is minimal. Explore:
- [ ] Mixup in feature space (interpolate between two patients' features + survival labels)
- [ ] CutMix on WSI bags (swap patches between patients)
- [ ] Pathway masking (randomly zero out entire pathways during training — different from modality dropout)
- **Motivation**: Small datasets (300-400 patients) benefit from augmentation

### 8E. Efficiency Analysis
Reviewers at ML venues care about computational cost:
- [ ] Report FLOPs and memory usage vs baselines (MCAT, SurvPath)
- [ ] Measure inference time per patient (ms)
- [ ] Show that Q-Former bottleneck reduces computation vs full cross-attention
- [ ] Parameter count comparison table

### 8F. WSI Encoder Comparison
Test sensitivity to the frozen encoder:
- [ ] UNI2-h (1536-d, current) — strongest
- [ ] CONCH (512-d) — pathology-specific CLIP
- [ ] CTransPath (768-d) — older but widely used
- [ ] ResNet-50 ImageNet (1024-d) — weakest, used by original SurvPath
- **Goal**: Show PathQ-Former architecture helps regardless of encoder quality

---

## Compute & Storage Budget

### Google Colab Free constraints
- **GPU**: T4 (15 GB VRAM) — sufficient for batch_size=1 + grad_accum
- **Session limit**: ~12 hours, may disconnect randomly
- **Disk**: ~110 GB local, ~15 GB Drive
- **Strategy**: Download embeddings each session (ephemeral), save checkpoints to Drive (persistent)

### Embedding storage per cancer
| Cancer | Size | Can fit on Colab disk? |
|--------|------|------------------------|
| BLCA | ~30 GB | Yes |
| BRCA | ~65 GB | Yes (barely, after extraction cleanup) |
| STAD | ~18 GB | Yes |
| COADREAD | ~22 GB | Yes |
| HNSC | ~17 GB | Yes |

### Time estimates (T4, per fold)
- ~10-15 min per epoch (BLCA, 423 slides)
- ~15 epochs average → ~2.5-4 hours per fold
- ~12-20 hours for 5-fold CV per cancer type
- Total for 5 cancers: ~60-100 hours of GPU time

---

## Published Baselines Reference

| Method | Venue | Modalities | Key Idea |
|--------|-------|------------|----------|
| ABMIL | ICML 2018 | WSI only | Attention-based MIL |
| MCAT | ICCV 2021 | WSI + genomics | Cross-attention between modalities |
| MOTCat | AAAI 2023 | WSI + genomics | Optimal transport + cross-attention |
| SurvPath | CVPR 2024 | WSI + pathways | Pathway-level tokenization + cross-attention |
| MMP | ICML 2024 | WSI + genomics | Masked multimodal pretraining |
| **PathQ-Former** | **Ours** | **WSI + pathways** | **Q-Former queries + missing-modality robustness** |

### BLCA C-Index ranges (DSS, from literature)
| Method | C-Index (approx) |
|--------|-------------------|
| ABMIL | ~0.52-0.55 |
| MCAT | ~0.58-0.61 |
| MOTCat | ~0.59-0.62 |
| SurvPath | ~0.60-0.63 |
| MMP | ~0.61-0.64 |
| **PathQ-Former (tuned, partial)** | **~0.68** (folds 0-1 mean) |

---

## Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Colab disconnects mid-fold | Lost partial training | Two-level resume system (fold + mid-epoch checkpoints) |
| BRCA embeddings too large for Colab disk | Can't run BRCA | Stream patches or subsample; or use Colab Pro |
| Tuned results don't generalize beyond BLCA | Weaker paper | Per-cancer hyperparameter search if needed |
| Baselines with UNI2-h perform equally well | No architectural advantage | Also report results with original encoders |
| NeurIPS workshop deadline too tight | Miss deadline | Aim for 3-cancer results minimum; full 5-cancer for ICLR |
| High fold variance (e.g., fold 2 weak) | Inflated std, weaker claim | Bootstrap CIs, report per-fold, analyze split difficulty |
| Reviewers ask for external validation | Major revision | CPTAC experiment (Phase 8A) as backup |
| C-Index alone not convincing | "Only ranking" criticism | Add td-AUC, Brier score, KM curves (Phase 5B) |
| Missing-modality story is thin | Weakens key claim | Simulate clinical scenarios with 10-50% missing rates |

---

## File Structure Reference

```
PathQFormer/
├── configs/              # YAML configs per experiment
│   ├── default.yaml
│   ├── blca.yaml
│   └── blca_tuned.yaml
├── src/
│   ├── models/           # PathQ-Former architecture
│   │   ├── pathqformer.py
│   │   ├── query_block.py
│   │   ├── fusion_block.py
│   │   ├── survival_head.py
│   │   └── null_tokens.py
│   ├── data/             # Dataset + pathway tokenizer
│   │   ├── tcga_dataset.py
│   │   └── pathway_tokenizer.py
│   └── training/         # Train loop + evaluation
│       ├── train.py
│       └── evaluate.py
├── notebooks/            # Colab notebooks
├── data/
│   ├── survpath_repo/    # SurvPath splits + metadata + RNA
│   └── embeddings/       # UNI2-h features (large, local only)
├── EXPERIMENT_LOG.md     # All results and observations
├── PLAN.md               # This file
└── PathQFormer_src.zip   # Packaged for Colab upload
```

---

## Priority Order (What to Do Next)

After Phase 2 finishes, this is the recommended sequence based on impact-per-hour:

1. **Phase 3** — Multi-cancer (required for any submission, ~1 week)
2. **Phase 5B** — Statistical rigor + extra metrics (cheap to add, high reviewer value)
3. **Phase 4E** — Component ablation (WSI-only, genomics-only, both — easy, high insight)
4. **Phase 5** — Missing-modality experiments (our key differentiator)
5. **Phase 6** — Baseline reproduction with UNI2-h (fair comparison, takes time)
6. **Phase 4A-D** — Full ablation sweep (important but time-intensive)
7. **Phase 7** — Paper writing (start once Tables 1-3 are populated)
8. **Phase 8** — Stretch goals (only if time permits before deadline)

---

## Changelog

| Date | Update |
|------|--------|
| 2026-05-19 | Created plan. Phase 1 complete, Phase 2 in progress. |
| 2026-05-19 | Expanded plan: added Phase 5B (statistical rigor), Phase 8 (extensions), ablations 4F-4G, priority order, additional tables/figures, risk mitigations. |
| 2026-09-10 | Protocol audit (6 defects), v2 pipeline with patient-level data, train-fit bins/scaler, continuous-time C-index, val-loss selection, seeds; Phase 5 + 5B metrics and Phase 4E ablations automated; pytest suite; first git commit; BLCA v2 queue launched on local CPU. |
| 2026-09-10 | Phase 6 started: official SurvPath + ABMIL + SNN + MLP baselines run through the same trainer (`model_type`), SurvPath-repo hyper-parameters, chained after the PathQ-Former queue. Analysis script (KM, curves, missing-modality plot, pathway attention) and generic v2 Colab notebook added. MCAT deferred (6-family gene groups not in the cloned data). |
