#!/usr/bin/env bash
# Regenerate every table in REPORT.md from the archived per-run results in pod_results/ (no GPU, ~1 min).
#
#   bash scripts/reproduce_tables.sh            # -> results/final/*.txt|md
#   PY=python3 bash scripts/reproduce_tables.sh # pick the interpreter
#
# The archives hold results.json / summary.md / config.yaml / per-fold metrics for all 180 GPU runs
# (checkpoints and figures are excluded; see README "Archived results").
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in .venv/Scripts/python.exe .venv/bin/python python3 python; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
  done
fi
echo "[tables] python: $PY"

# Unpack the archives on a fresh clone (the extracted directories are git-ignored).
if [ ! -f pod_results/outputs_e20/efficiency.md ] || [ ! -d pod_results/outputs_os ]; then
  echo "[tables] extracting archived results"
  for t in pod_results/pod_results.tgz pod_results/pull_0918/results_0918a.tgz pod_results/pull_0918/results_0918b.tgz pod_results/pull_0923/results_0923.tgz; do
    tar xzf "$t" -C pod_results/
  done
fi

OUT=results/final
mkdir -p "$OUT"
E20=pod_results/outputs_e20

echo "[tables] Table 1: DSS, 20 epochs, 3 seeds, reference SurvPath"
"$PY" scripts/seed_table.py "$E20" --methods pathq_fast_e20 pathq_fast_e20_aux survpath_e20 abmil_e20 snn_e20 mlp_omics_e20 \
  --ref survpath_e20 > "$OUT/table1_dss_20ep_vs_survpath.txt"

echo "[tables] Table 2: same runs, reference RNA MLP"
"$PY" scripts/seed_table.py "$E20" --methods pathq_fast_e20 pathq_fast_e20_aux survpath_e20 mlp_omics_e20 \
  --ref mlp_omics_e20 > "$OUT/table2_dss_20ep_vs_mlp.txt"

echo "[tables] OS endpoint"
"$PY" scripts/seed_table.py pod_results/outputs_os --methods pathq_fast_e20_aux survpath_e20 --ref survpath_e20 > "$OUT/table_os_20ep.txt"

echo "[tables] 10-epoch budget (seed 0 in outputs_v2, seeds 1-2 in outputs_ablate)"
"$PY" scripts/seed_table.py pod_results/outputs_v2 pod_results/outputs_ablate --methods pathq_fast_e10 survpath_e10 \
  --ref survpath_e10 > "$OUT/table_dss_10ep.txt"

echo "[tables] per-run summaries (all metrics, missing-modality blocks, per-fold rows)"
"$PY" scripts/aggregate_results.py "$E20" --out "$OUT/summary_e20_all_runs.md" >/dev/null
"$PY" scripts/aggregate_results.py pod_results/outputs_os --out "$OUT/summary_os_all_runs.md" >/dev/null
"$PY" scripts/aggregate_results.py pod_results/outputs_ablate --out "$OUT/summary_ablate_all_runs.md" >/dev/null
"$PY" scripts/aggregate_results.py pod_results/outputs_v2 --out "$OUT/summary_v2_all_runs.md" >/dev/null

echo "[tables] fusion vs late-fusion ensemble (seeds available), grid"
"$PY" scripts/seed_table.py "$E20" --methods pathq_fast_e20_aux pathq_fast_e20 late_fusion_e20 pathq_e20_wsi_only pathq_e20_genomic_only \
  --ref late_fusion_e20 > "$OUT/table_fusion_vs_ensemble.txt"
[ -d pod_results/outputs_grid ] && "$PY" scripts/seed_table.py pod_results/outputs_grid "$E20" \
  --methods pathq_fast_e20_aux $(ls pod_results/outputs_grid 2>/dev/null | grep -v summary | tr "\n" " ") survpath_e20 --ref survpath_e20 > "$OUT/table_grid.txt" || true

echo "[tables] missing-modality over seeds, risk correlations"
"$PY" scripts/missing_modality_table.py "$E20" --methods pathq_fast_e20_aux pathq_fast_e20 survpath_e20 > "$OUT/table_missing_modality_seeds.txt"   # incl. *_impute and rna/wsi_permuted
"$PY" scripts/risk_correlation.py "$E20" --run pathq_fast_e20_aux --others survpath_e20 mlp_omics_e20 abmil_e20 snn_e20 pathq_fast_e20 > "$OUT/table_risk_correlation.txt"

echo "[tables] post-hoc checkpoint-selection rules replayed on the stored per-epoch histories (selects on the held-out fold)"
"$PY" scripts/posthoc_selection.py --roots "$E20" pod_results/outputs_v2 pod_results/outputs_ablate --v2-root pod_results/outputs_v2 \
  --patience 5 --min-epochs 0 --out - > "$OUT/table_selection_rules.txt"

echo "[tables] primary seed-averaged paired tests (bootstrap CI, Holm over cohorts) and secondary metrics (IPCW C, IBS)"
"$PY" scripts/primary_tests.py --e20 "$E20" --os pod_results/outputs_os --out-dir "$OUT" >/dev/null   # -> table_primary_tests.txt, table_secondary_metrics.txt

cp "$E20/efficiency.md" "$OUT/efficiency.md"

echo "[tables] done -> $OUT/"
ls -1 "$OUT"
