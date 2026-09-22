# Compute cost per patient (cuda, BLCA fold-0 validation, 40 patients, medians)

| Model | Params (M) | Patches | Forward (ms) | Forward+backward (ms) | Peak GPU mem (GiB) |
|---|---|---|---|---|---|
| PathQ-Former (all patches) | 16.9 | 6214 | 35.0 | 104.2 | 1.17 |
| PathQ-Former (4096 patches) | 16.9 | 4096 | 34.7 | 104.1 | 0.17 |
| SurvPath (4096 patches, as trained) | 21.2 | 4096 | 35.7 | 175.6 | 0.20 |
| SurvPath (all patches) | 21.2 | 6214 | 36.0 | 176.0 | 1.74 |
