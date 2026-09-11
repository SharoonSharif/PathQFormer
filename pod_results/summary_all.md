# PathQ-Former results

19 run(s) under pod_results/outputs_v2

| Run | Cancer | Folds | Train modalities | Selection | C-index (both) | 95% CI | IPCW | IBS | td-AUC | WSI-only | Genomics-only | log-rank p (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_abmil | BRCA | 5 | wsi | val_loss | **0.4813 ± 0.1137** | [0.340, 0.623] | 0.5105 | 0.1153 | 0.5264 | 0.4813 | nan | 0.363 |
| baseline_abmil | HNSC | 5 | wsi | val_loss | **0.5467 ± 0.0451** | [0.491, 0.603] | 0.5650 | 0.2292 | 0.5471 | 0.5467 | nan | 0.449 |
| baseline_abmil | STAD | 5 | wsi | val_loss | **0.4975 ± 0.0889** | [0.387, 0.608] | 0.4627 | 0.2207 | 0.4697 | 0.4975 | nan | 0.235 |
| baseline_mlp_omics | BRCA | 5 | genomic | val_loss | **0.5393 ± 0.1134** | [0.399, 0.680] | 0.5555 | 0.0844 | 0.6251 | nan | 0.5393 | 0.473 |
| baseline_mlp_omics | HNSC | 5 | genomic | val_loss | **0.5429 ± 0.0166** | [0.522, 0.563] | 0.5428 | 0.2282 | 0.5397 | nan | 0.5429 | 0.764 |
| baseline_mlp_omics | STAD | 5 | genomic | val_loss | **0.5156 ± 0.0466** | [0.458, 0.573] | 0.5400 | 0.2177 | 0.5499 | nan | 0.5156 | 0.261 |
| baseline_snn | BRCA | 5 | genomic | val_loss | **0.5719 ± 0.1068** | [0.439, 0.705] | 0.5805 | 0.0863 | 0.6465 | nan | 0.5719 | 0.217 |
| baseline_snn | HNSC | 5 | genomic | val_loss | **0.4715 ± 0.0658** | [0.390, 0.553] | 0.4716 | 0.2222 | 0.5106 | nan | 0.4715 | 0.535 |
| baseline_snn | STAD | 5 | genomic | val_loss | **0.4535 ± 0.1156** | [0.310, 0.597] | 0.4573 | 0.2260 | 0.5113 | nan | 0.4535 | 0.323 |
| baseline_survpath | BRCA | 5 | both | val_loss | **0.5318 ± 0.1837** | [0.304, 0.760] | 0.5364 | 0.1754 | 0.5587 | nan | nan | 0.313 |
| baseline_survpath | HNSC | 5 | both | val_loss | **0.5153 ± 0.0568** | [0.445, 0.586] | 0.5181 | 0.1985 | 0.5349 | nan | nan | 0.215 |
| baseline_survpath | STAD | 5 | both | val_loss | **0.6097 ± 0.0744** | [0.517, 0.702] | 0.5964 | 0.1986 | 0.6344 | nan | nan | 0.118 |
| baseline_survpath_seed1 | BLCA | 5 | both | val_loss | **0.5520 ± 0.1421** | [0.376, 0.728] | 0.5339 | 0.1778 | 0.6020 | nan | nan | 0.121 |
| baseline_survpath_seed2 | BLCA | 5 | both | val_loss | **0.5974 ± 0.0468** | [0.539, 0.656] | 0.5925 | 0.1783 | 0.6290 | nan | nan | 0.442 |
| hybrid | BRCA | 5 | both | val_loss | **0.5475 ± 0.1384** | [0.376, 0.719] | 0.5320 | 0.0602 | 0.5509 | 0.5403 | 0.5861 | 0.661 |
| hybrid | HNSC | 5 | both | val_loss | **0.4839 ± 0.0836** | [0.380, 0.588] | 0.4966 | 0.1496 | 0.4716 | 0.4670 | 0.5130 | 0.698 |
| hybrid | STAD | 5 | both | val_loss | **0.4860 ± 0.0688** | [0.401, 0.571] | 0.4932 | 0.1506 | 0.4657 | 0.4754 | 0.5547 | 0.409 |
| hybrid_seed1 | BLCA | 5 | both | val_loss | **0.5989 ± 0.1316** | [0.436, 0.762] | 0.5964 | 0.1521 | 0.6561 | 0.5780 | 0.5730 | 0.163 |
| hybrid_seed2 | BLCA | 5 | both | val_loss | **0.6220 ± 0.1046** | [0.492, 0.752] | 0.6209 | 0.1500 | 0.6659 | 0.5747 | 0.5825 | 0.779 |

### Random missing modality at test time (C-index, mean over folds)

| Run | Cancer | 0% | 10% | 20% | 30% | 50% | WSI-only (100%) | Genomics-only (100%) |
|---|---|---|---|---|---|---|---|---|
| hybrid | BRCA | 0.5475 | 0.5326 | 0.5233 | 0.5435 | 0.5231 | 0.5403 | 0.5861 |
| hybrid | HNSC | 0.4839 | 0.4955 | 0.4845 | 0.5087 | 0.5075 | 0.4670 | 0.5130 |
| hybrid | STAD | 0.4860 | 0.4899 | 0.5047 | 0.5148 | 0.4963 | 0.4754 | 0.5547 |
| hybrid_seed1 | BLCA | 0.5989 | 0.5899 | 0.5938 | 0.6003 | 0.5900 | 0.5780 | 0.5730 |
| hybrid_seed2 | BLCA | 0.6220 | 0.6225 | 0.6211 | 0.6064 | 0.6020 | 0.5747 | 0.5825 |

### Per-fold C-index (selected checkpoint)

| Run | Cancer | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|---|
| baseline_abmil | BRCA | 0.4347 | 0.5491 | 0.5710 | 0.3018 | 0.5499 | 0.4813 |
| baseline_abmil | HNSC | 0.4906 | 0.6120 | 0.5321 | 0.5339 | 0.5648 | 0.5467 |
| baseline_abmil | STAD | 0.3444 | 0.5751 | 0.5194 | 0.5346 | 0.5141 | 0.4975 |
| baseline_mlp_omics | BRCA | 0.5498 | 0.3428 | 0.5868 | 0.5881 | 0.6291 | 0.5393 |
| baseline_mlp_omics | HNSC | 0.5254 | 0.5586 | 0.5468 | 0.5580 | 0.5256 | 0.5429 |
| baseline_mlp_omics | STAD | 0.5066 | 0.4577 | 0.5347 | 0.4962 | 0.5827 | 0.5156 |
| baseline_snn | BRCA | 0.5928 | 0.3926 | 0.5689 | 0.6542 | 0.6512 | 0.5719 |
| baseline_snn | HNSC | 0.4456 | 0.3983 | 0.5389 | 0.5432 | 0.4317 | 0.4715 |
| baseline_snn | STAD | 0.3046 | 0.3921 | 0.6160 | 0.4769 | 0.4778 | 0.4535 |
| baseline_survpath | BRCA | 0.4433 | 0.5434 | 0.7024 | 0.2687 | 0.7014 | 0.5318 |
| baseline_survpath | HNSC | 0.5283 | 0.5595 | 0.4171 | 0.5240 | 0.5478 | 0.5153 |
| baseline_survpath | STAD | 0.6755 | 0.6149 | 0.6042 | 0.6654 | 0.4886 | 0.6097 |
| baseline_survpath_seed1 | BLCA | 0.5317 | 0.3868 | 0.7234 | 0.4504 | 0.6676 | 0.5520 |
| baseline_survpath_seed2 | BLCA | 0.5692 | 0.5822 | 0.6045 | 0.5562 | 0.6748 | 0.5974 |
| hybrid | BRCA | 0.6031 | 0.6586 | 0.5124 | 0.3216 | 0.6420 | 0.5475 |
| hybrid | HNSC | 0.4180 | 0.4199 | 0.6144 | 0.4486 | 0.5188 | 0.4839 |
| hybrid | STAD | 0.3874 | 0.4957 | 0.4912 | 0.5808 | 0.4751 | 0.4860 |
| hybrid_seed1 | BLCA | 0.5885 | 0.4894 | 0.6926 | 0.4570 | 0.7669 | 0.5989 |
| hybrid_seed2 | BLCA | 0.5846 | 0.6451 | 0.6875 | 0.4618 | 0.7309 | 0.6220 |

### Paired tests vs `hybrid` (same folds)

| Run | Cancer | delta C-index | paired t p | Wilcoxon p | n folds |
|---|---|---|---|---|---|
| baseline_abmil | BRCA | -0.0663 | 0.166 | 0.188 | 5 |
| baseline_abmil | HNSC | 0.0627 | 0.227 | 0.312 | 5 |
| baseline_abmil | STAD | 0.0115 | 0.663 | 1.000 | 5 |
| baseline_mlp_omics | BRCA | -0.0082 | 0.935 | 1.000 | 5 |
| baseline_mlp_omics | HNSC | 0.0589 | 0.203 | 0.188 | 5 |
| baseline_mlp_omics | STAD | 0.0295 | 0.501 | 0.438 | 5 |
| baseline_snn | BRCA | 0.0244 | 0.811 | 0.812 | 5 |
| baseline_snn | HNSC | -0.0124 | 0.731 | 1.000 | 5 |
| baseline_snn | STAD | -0.0325 | 0.500 | 0.812 | 5 |
| baseline_survpath | BRCA | -0.0157 | 0.816 | 1.000 | 5 |
| baseline_survpath | HNSC | 0.0314 | 0.629 | 0.625 | 5 |
| baseline_survpath | STAD | 0.1237 | 0.052 | 0.062 | 5 |
