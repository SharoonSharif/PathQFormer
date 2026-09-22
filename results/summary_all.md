# PathQ-Former results

13 run(s) under outputs_v2

| Run | Cancer | Folds | Train modalities | Selection | C-index (both) | 95% CI | IPCW | IBS | td-AUC | WSI-only | Genomics-only | log-rank p (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | BLCA | 5 | both | val_loss | **0.5744 ± 0.0949** | [0.456, 0.692] | 0.5655 | 0.1815 | 0.5964 | 0.5504 | 0.5915 | 0.185 |
| baseline_abmil | BLCA | 5 | wsi | val_loss | **0.5653 ± 0.0702** | [0.478, 0.652] | 0.5564 | 0.1978 | 0.5831 | 0.5653 | nan | 0.853 |
| baseline_mlp_omics | BLCA | 5 | genomic | val_loss | **0.5714 ± 0.0793** | [0.473, 0.670] | 0.5642 | 0.2014 | 0.6166 | nan | 0.5714 | 0.468 |
| baseline_snn | BLCA | 5 | genomic | val_loss | **0.4985 ± 0.0575** | [0.427, 0.570] | 0.4960 | 0.2108 | 0.5931 | nan | 0.4985 | 0.528 |
| baseline_survpath | BLCA | 5 | both | val_loss | **0.5592 ± 0.1193** | [0.411, 0.707] | 0.5628 | 0.1756 | 0.5713 | nan | nan | 0.314 |
| genomic_only | BLCA | 5 | genomic | val_loss | **0.6463 ± 0.0696** | [0.560, 0.733] | 0.6279 | 0.1526 | 0.6492 | nan | 0.6463 | 0.324 |
| hybrid | BLCA | 5 | both | val_loss | **0.6447 ± 0.0859** | [0.538, 0.751] | 0.6243 | 0.1546 | 0.7082 | 0.5906 | 0.6283 | 0.134 |
| pathq_aux_e20_bins2 | BLCA | 5 | both | last | **0.6278 ± 0.0567** | [0.557, 0.698] | 0.6433 | nan | 0.6147 | 0.6339 | 0.5755 | 0.211 |
| pathq_aux_e20_bins8 | BLCA | 5 | both | last | **0.6093 ± 0.0815** | [0.508, 0.711] | 0.5962 | 0.1935 | 0.6444 | 0.5785 | 0.6054 | 0.359 |
| pathq_aux_e20_hallmarks | BLCA | 5 | both | last | **0.6000 ± 0.0753** | [0.507, 0.693] | 0.6053 | 0.2222 | 0.6733 | 0.6078 | 0.5544 | 0.373 |
| pathq_aux_e20_patches4096 | BLCA | 5 | both | last | **0.6264 ± 0.0591** | [0.553, 0.700] | 0.6291 | 0.2078 | 0.6876 | 0.5912 | 0.6221 | 0.028 |
| pathq_aux_e20_xena | BLCA | 5 | both | last | **0.6283 ± 0.0625** | [0.551, 0.706] | 0.6241 | 0.2238 | 0.6655 | 0.6038 | 0.6339 | 0.112 |
| wsi_only | BLCA | 5 | wsi | val_loss | **0.6183 ± 0.0426** | [0.565, 0.671] | 0.6202 | 0.1575 | 0.6669 | 0.6183 | nan | 0.131 |

### Random missing modality at test time (C-index, mean over folds)

| Run | Cancer | 0% | 10% | 20% | 30% | 50% | WSI-only (100%) | Genomics-only (100%) |
|---|---|---|---|---|---|---|---|---|
| baseline | BLCA | 0.5744 | 0.5700 | 0.5554 | 0.5631 | 0.5408 | 0.5504 | 0.5915 |
| hybrid | BLCA | 0.6447 | 0.6473 | 0.6390 | 0.6209 | 0.6233 | 0.5906 | 0.6283 |
| pathq_aux_e20_bins2 | BLCA | 0.6278 | 0.6266 | 0.6279 | 0.6204 | 0.6161 | 0.6339 | 0.5755 |
| pathq_aux_e20_bins8 | BLCA | 0.6093 | 0.6172 | 0.6192 | 0.6105 | 0.6043 | 0.5785 | 0.6054 |
| pathq_aux_e20_hallmarks | BLCA | 0.6000 | 0.5998 | 0.6060 | 0.5964 | 0.5972 | 0.6078 | 0.5544 |
| pathq_aux_e20_patches4096 | BLCA | 0.6264 | 0.6269 | 0.6254 | 0.6209 | 0.6112 | 0.5912 | 0.6221 |
| pathq_aux_e20_xena | BLCA | 0.6283 | 0.6307 | 0.6287 | 0.6256 | 0.6146 | 0.6038 | 0.6339 |

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
| pathq_aux_e20_bins2 | BLCA | 0.6096 | 0.5967 | 0.6936 | 0.5600 | 0.6791 | 0.6278 |
| pathq_aux_e20_bins8 | BLCA | 0.6837 | 0.5629 | 0.6711 | 0.4901 | 0.6388 | 0.6093 |
| pathq_aux_e20_hallmarks | BLCA | 0.6269 | 0.5629 | 0.7172 | 0.5231 | 0.5698 | 0.6000 |
| pathq_aux_e20_patches4096 | BLCA | 0.6375 | 0.6161 | 0.6670 | 0.5307 | 0.6806 | 0.6264 |
| pathq_aux_e20_xena | BLCA | 0.6067 | 0.6567 | 0.6516 | 0.5316 | 0.6950 | 0.6283 |
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
| pathq_aux_e20_bins2 | BLCA | -0.0169 | 0.401 | 0.625 | 5 |
| pathq_aux_e20_bins8 | BLCA | -0.0354 | 0.346 | 0.438 | 5 |
| pathq_aux_e20_hallmarks | BLCA | -0.0448 | 0.314 | 0.625 | 5 |
| pathq_aux_e20_patches4096 | BLCA | -0.0184 | 0.370 | 0.438 | 5 |
| pathq_aux_e20_xena | BLCA | -0.0164 | 0.360 | 0.625 | 5 |
| wsi_only | BLCA | -0.0265 | 0.479 | 0.312 | 5 |
