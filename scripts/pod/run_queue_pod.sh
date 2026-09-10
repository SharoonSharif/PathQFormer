#!/usr/bin/env bash
# Unattended multi-cancer experiment queue for a Runpod GPU pod (network volume at /workspace).
#
#   cd /workspace/PathQFormer
#   setsid bash scripts/pod/run_queue_pod.sh > /workspace/logs/pod_queue.out 2>&1 < /dev/null &
#
# Per cohort (small ones first so results arrive early): stream the embeddings into the volume,
# then PathQ-Former (hybrid v2) -> official SurvPath -> ABMIL -> SNN -> MLP, all under the identical
# protocol. Then extra seeds on BLCA for the two multimodal methods (seed 0 runs on the laptop).
# Everything resumes (finished folds are skipped), so re-running this script is always safe.
# When done: recompute metrics, build the comparison table, render figures, and remove the pod
# itself unless KEEP_POD=1.
set -uo pipefail
cd /workspace/PathQFormer
set -a; source /workspace/.env; set +a
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-${RUNPOD_API_KEY_PASTED:-}}" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
OUT="${OUT:-/workspace/outputs_v2}"; EMB="${EMBEDDINGS_ROOT:-/workspace/embeddings}"
COHORTS="${COHORTS:-STAD HNSC COADREAD BRCA}"
SEEDS="${SEEDS:-1 2}"
WORKERS="${WORKERS:-6}"
PY=python3

run() {  # run <config> <cancer_lower> <run_name> [extra overrides "k=v,k=v"]
  local cfg=$1 cancer=$2 name=$3 extra=${4:-}
  bash scripts/run_experiments.sh \
    "${cfg}::cancer_type=${cancer},embeddings_dir=${EMB}/${cancer^^},output_dir=${OUT}/${name},num_workers=${WORKERS}${extra:+,$extra}"
}

echo "[queue] $(date '+%F %T') start | cohorts: $COHORTS | extra BLCA seeds: $SEEDS | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
for C in $COHORTS; do
  if ! bash scripts/pod/fetch_cohort.sh "$C"; then echo "[queue] $C: fetch failed, skipping"; continue; fi
  c=${C,,}
  run configs/blca_hybrid_v2.yaml            "$c" hybrid
  run configs/baselines/blca_survpath.yaml   "$c" baseline_survpath
  run configs/baselines/blca_abmil.yaml      "$c" baseline_abmil
  run configs/baselines/blca_snn.yaml        "$c" baseline_snn
  run configs/baselines/blca_mlp_omics.yaml  "$c" baseline_mlp_omics
done

if bash scripts/pod/fetch_cohort.sh BLCA; then
  for s in $SEEDS; do
    run configs/blca_hybrid_v2.yaml          blca "hybrid_seed$s"            "seed=$s"
    run configs/baselines/blca_survpath.yaml blca "baseline_survpath_seed$s" "seed=$s"
  done
fi

echo "[queue] $(date '+%F %T') training done; finalizing"
finished=()
for r in "$OUT"/*/*/results.json; do [ -f "$r" ] && finished+=("$(dirname "$r")"); done
[ "${#finished[@]}" -gt 0 ] && $PY scripts/recompute_metrics.py "${finished[@]}"
$PY scripts/aggregate_results.py "$OUT" --out "$OUT/summary_all.md" --compare hybrid
for r in "${finished[@]}"; do $PY scripts/analyze_run.py "$r" >/dev/null 2>&1 || echo "[queue] analysis failed for $r"; done
echo "[queue] $(date '+%F %T') all done -> $OUT/summary_all.md"

if [ "${KEEP_POD:-0}" != "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  echo "[queue] removing this pod ($RUNPOD_POD_ID); the volume keeps everything"
  runpodctl pod remove "$RUNPOD_POD_ID"
fi
