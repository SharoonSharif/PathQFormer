# PathQ-Former results

10 run(s) under pod_results/outputs_os

| Run | Cancer | Folds | Train modalities | Selection | C-index (both) | 95% CI | IPCW | IBS | td-AUC | WSI-only | Genomics-only | log-rank p (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pathq_fast_e20_aux | BLCA | 5 | both | last | **0.5954 ± 0.0532** | [0.529, 0.661] | 0.5868 | 0.2810 | 0.6639 | 0.5758 | 0.5917 | 0.268 |
| pathq_fast_e20_aux | BRCA | 5 | both | last | **0.5963 ± 0.0427** | [0.543, 0.649] | 0.5521 | 0.1212 | 0.5402 | 0.5309 | 0.5558 | 0.154 |
| pathq_fast_e20_aux | COADREAD | 5 | both | last | **0.5808 ± 0.1316** | [0.417, 0.744] | 0.5884 | 0.1782 | 0.6080 | 0.5863 | 0.5620 | 0.345 |
| pathq_fast_e20_aux | HNSC | 5 | both | last | **0.5358 ± 0.0364** | [0.491, 0.581] | 0.5366 | 0.3114 | 0.5215 | 0.5347 | 0.5502 | 0.806 |
| pathq_fast_e20_aux | STAD | 5 | both | last | **0.6029 ± 0.0698** | [0.516, 0.690] | 0.6074 | 0.2774 | 0.6254 | 0.5740 | 0.5818 | 0.222 |
| survpath_e20 | BLCA | 5 | both | last | **0.5521 ± 0.0559** | [0.483, 0.622] | 0.5550 | 0.3632 | 0.6165 | nan | nan | 0.536 |
| survpath_e20 | BRCA | 5 | both | last | **0.5234 ± 0.0517** | [0.459, 0.588] | 0.5627 | 0.2245 | 0.5453 | nan | nan | 0.341 |
| survpath_e20 | COADREAD | 5 | both | last | **0.5960 ± 0.1530** | [0.406, 0.786] | 0.5828 | 0.2560 | 0.4625 | nan | nan | 0.409 |
| survpath_e20 | HNSC | 5 | both | last | **0.5104 ± 0.0288** | [0.475, 0.546] | 0.5181 | 0.3575 | 0.4984 | nan | nan | 0.902 |
| survpath_e20 | STAD | 5 | both | last | **0.5590 ± 0.0451** | [0.503, 0.615] | 0.5671 | 0.3026 | 0.5566 | nan | nan | 0.374 |

### Random missing modality at test time (C-index, mean over folds)

| Run | Cancer | 0% | 10% | 20% | 30% | 50% | WSI-only (100%) | Genomics-only (100%) |
|---|---|---|---|---|---|---|---|---|
| pathq_fast_e20_aux | BLCA | 0.5954 | 0.5974 | 0.6056 | 0.5969 | 0.5905 | 0.5758 | 0.5917 |
| pathq_fast_e20_aux | BRCA | 0.5963 | 0.5900 | 0.5901 | 0.5797 | 0.5757 | 0.5309 | 0.5558 |
| pathq_fast_e20_aux | COADREAD | 0.5808 | 0.5782 | 0.5739 | 0.5803 | 0.5759 | 0.5863 | 0.5620 |
| pathq_fast_e20_aux | HNSC | 0.5358 | 0.5379 | 0.5381 | 0.5390 | 0.5310 | 0.5347 | 0.5502 |
| pathq_fast_e20_aux | STAD | 0.6029 | 0.6002 | 0.6009 | 0.6051 | 0.5912 | 0.5740 | 0.5818 |

### Per-fold C-index (selected checkpoint)

| Run | Cancer | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|---|---|
| pathq_fast_e20_aux | BLCA | 0.5778 | 0.6499 | 0.6016 | 0.5144 | 0.6331 | 0.5954 |
| pathq_fast_e20_aux | BRCA | 0.5609 | 0.5461 | 0.6294 | 0.5998 | 0.6453 | 0.5963 |
| pathq_fast_e20_aux | COADREAD | 0.7461 | 0.4307 | 0.5436 | 0.6860 | 0.4974 | 0.5808 |
| pathq_fast_e20_aux | HNSC | 0.5401 | 0.5680 | 0.4975 | 0.5739 | 0.4994 | 0.5358 |
| pathq_fast_e20_aux | STAD | 0.5795 | 0.6140 | 0.6760 | 0.6488 | 0.4962 | 0.6029 |
| survpath_e20 | BLCA | 0.5469 | 0.6288 | 0.5533 | 0.4713 | 0.5605 | 0.5521 |
| survpath_e20 | BRCA | 0.5368 | 0.4731 | 0.5331 | 0.4755 | 0.5983 | 0.5234 |
| survpath_e20 | COADREAD | 0.8398 | 0.4827 | 0.5436 | 0.6453 | 0.4688 | 0.5960 |
| survpath_e20 | HNSC | 0.5033 | 0.5423 | 0.5060 | 0.5320 | 0.4685 | 0.5104 |
| survpath_e20 | STAD | 0.5464 | 0.6330 | 0.5344 | 0.5652 | 0.5161 | 0.5590 |
