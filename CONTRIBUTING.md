# Contributing

Thanks for your interest. The project is small enough that the whole workflow fits on one page.

## Set up

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
uv pip install --python .venv/bin/python -r requirements.txt einops
git clone --depth 1 https://github.com/mahmoodlab/SurvPath data/survpath_repo    # splits, metadata, RNA, pathways
python scripts/make_dummy_embeddings.py --cancer blca                            # random features so tests run without TCGA data
python -m pytest -m "not slow"                                                   # ~20 s
```

On Windows replace `.venv/bin/python` with `.venv/Scripts/python.exe`. `make install dummy-data test` does the same.

## Before opening a pull request

- `python -m pytest` (the slow end-to-end trainer tests take ~10 min on CPU; CI runs them on request).
- `ruff check .` (line length 120, configured in `pyproject.toml`).
- If you change anything that affects a reported number (data loading, metrics, bins, selection), say so in
  the PR and add a line to `EXPERIMENT_LOG.md`. `scripts/recompute_metrics.py` re-derives every metric block
  from the saved per-patient predictions, so metric changes can be applied to archived runs.
- New baselines go in `src/models/baselines.py` and the `BASELINES` dict; new configs in `configs/`;
  every model must implement `supports_missing` and the `forward(wsi_features, genomic_features, wsi_mask)`
  signature so the trainer and the missing-modality evaluation work unchanged.

## Reporting results

Every run writes `results.json` (per-fold metrics, seeds, git commit, config) and `summary.md`. Cross-run
tables with paired tests come from `scripts/seed_table.py` and `scripts/aggregate_results.py`; please report
the fixed-budget protocol (`selection_metric: last`) and at least three seeds when comparing methods, for the
reasons documented in `REPORT.md` section 3.

## Questions

Open an issue. Please do not attach TCGA data or gated UNI2-h features to issues or PRs.
