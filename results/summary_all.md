# PathQ-Former results

8 run(s) under outputs_v2

| Run | Cancer | Folds | Train modalities | Selection | C-index (both) | 95% CI | IPCW | IBS | td-AUC | WSI-only | Genomics-only | log-rank p (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | BLCA | 5 | both | val_loss | **0.5744 ± 0.0949** | [0.456, 0.692] | 0.5655 | 0.1815 | 0.5964 | 0.5504 | 0.5915 | 0.185 |
| baseline_abmil | BLCA | 5 | wsi | val_loss | **0.5653 ± 0.0702** | [0.478, 0.652] | 0.5564 | 0.1978 | 0.5831 | 0.5653 | nan | 0.853 |
| baseline_mlp_omics | BLCA | 5 | genomic | val_loss | **0.5714 ± 0.0793** | [0.473, 0.670] | 0.5642 | 0.2014 | 0.6166 | nan | 0.5714 | 0.468 |
| baseline_snn | BLCA | 5 | genomic | val_loss | **0.4985 ± 0.0575** | [0.427, 0.570] | 0.4960 | 0.2108 | 0.5931 | nan | 0.4985 | 0.528 |
| baseline_survpath | BLCA | 5 | both | val_loss | **0.5592 ± 0.1193** | [0.411, 0.707] | 0.5628 | 0.1756 | 0.5713 | nan | nan | 0.314 |
| genomic_only | BLCA | 5 | genomic | val_loss | **0.6463 ± 0.0696** | [0.560, 0.733] | 0.6279 | 0.1526 | 0.6492 | nan | 0.6463 | 0.324 |
| hybrid | BLCA | 5 | both | val_loss | **0.6447 ± 0.0859** | [0.538, 0.751] | 0.6243 | 0.1546 | 0.7082 | 0.5906 | 0.6283 | 0.134 |
| wsi_only | BLCA | 5 | wsi | val_loss | **0.6183 ± 0.0426** | [0.565, 0.671] | 0.6202 | 0.1575 | 0.6669 | 0.6183 | nan | 0.131 |

### Random missing modality at test time (C-index, mean over folds)

| Run | Cancer | 0% | 10% | 20% | 30% | 50% | WSI-only (100%) | Genomics-only (100%) |
|---|---|---|---|---|---|---|---|---|
| baseline | BLCA | 0.5744 | 0.5700 | 0.5554 | 0.5631 | 0.5408 | 0.5504 | 0.5915 |
| hybrid | BLCA | 0.6447 | 0.6473 | 0.6390 | 0.6209 | 0.6233 | 0.5906 | 0.6283 |

### Per-fold C-index (selected checkpoint)

| Run | Cancer | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|---|
| baseline | BLCA | 0.6212 | 0.6088 | 0.5871 | 0.4089 | 0.6460 | 0.5744 |
| baseline_abmil | BLCA | 0.5058 | 0.5445 | 0.5338 | 0.5562 | 0.6863 | 0.5653 |
| baseline_mlp_omics | BLCA | 0.5529 | 0.5145 | 0.7008 | 0.5033 | 0.5856 | 0.5714 |
| baseline_snn | BLCA | 0.5308 | 0.4159 | 0.4662 | 0.5184 | 0.5612 | 0.4985 |
| baseline_survpath | BLCA | 0.4673 | 0.5919 | 0.5635 | 0.4353 | 0.7381 | 0.5592 |
| genomic_only | BLCA | 0.6067 | 0.6306 | 0.7377 | 0.5628 | 0.6935 | 0.6463 |
| hybrid | BLCA | 0.6000 | 0.6354 | 0.7008 | 0.5335 | 0.7540 | 0.6447 |
| wsi_only | BLCA | 0.6615 | 0.5706 | 0.6137 | 0.5836 | 0.6619 | 0.6183 |

### Paired tests vs `hybrid` (same folds)

| Run | Cancer | delta C-index | paired t p | Wilcoxon p | n folds |
|---|---|---|---|---|---|
| baseline | BLCA | -0.0703 | 0.071 | 0.125 | 5 |
| baseline_abmil | BLCA | -0.0794 | 0.060 | 0.125 | 5 |
| baseline_mlp_omics | BLCA | -0.0733 | 0.077 | 0.125 | 5 |
| baseline_snn | BLCA | -0.1463 | 0.029 | 0.062 | 5 |
| baseline_survpath | BLCA | -0.0855 | 0.024 | 0.062 | 5 |
| genomic_only | BLCA | 0.0015 | 0.934 | 0.812 | 5 |
| wsi_only | BLCA | -0.0265 | 0.479 | 0.312 | 5 |
