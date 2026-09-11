#!/usr/bin/env bash
# Unattended experiment queue for a Runpod GPU pod (network volume at /workspace).
#
#   cd /workspace/PathQFormer
#   setsid bash scripts/pod/run_queue_pod.sh > /workspace/logs/pod_queue.out 2>&1 < /dev/null &
#
# For every cohort in COHORTS (embeddings streamed into the volume on first use), every seed in SEEDS
# and every config in CONFIGS, run the training under the identical protocol. Everything resumes
# (finished folds are skipped), so re-running is always safe. Afterwards: recompute metrics, build the
# comparison table, render figures, and remove the pod itself unless KEEP_POD=1.
#
# Environment knobs (defaults in brackets):
#   COHORTS  [STAD HNSC COADREAD BRCA]   SEEDS [0]   WORKERS [6]   OUT [/workspace/outputs_v2]
#   CONFIGS  [PathQ-Former hybrid + SurvPath + ABMIL + SNN + MLP baselines]
#   CACHE_COHORTS [BLCA STAD HNSC COADREAD]  cohorts whose features are cached in RAM as float16
#            (epochs after the first become compute-bound; BRCA's 66 GB is left on disk)
#   EXTRA    []  extra "key=value,key=value" overrides appended to every run
#   Run names come from the config file name: blca_hybrid_v2 -> hybrid, blca_survpath -> baseline_survpath,
#   anything else -> its own stem (e.g. pathq_fast_e10); seeds other than 0 add "_seed<N>".
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the repo this script lives in
set -a; source "${ENV_FILE:-/workspace/.env}"; set +a
[ -f /etc/rp_environment ] && source /etc/rp_environment   # RUNPOD_POD_ID etc. of THIS pod (overrides any id in .env)
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-${RUNPOD_API_KEY_PASTED:-}}" PYTHONUTF8=1 PYTHONIOENCODING=utf-8
OUT="${OUT:-/workspace/outputs_v2}"; EMB="${EMBEDDINGS_ROOT:-/workspace/embeddings}"
COHORTS="${COHORTS:-STAD HNSC COADREAD BRCA}"
SEEDS="${SEEDS:-0}"
WORKERS="${WORKERS:-6}"
CACHE_COHORTS="${CACHE_COHORTS:-BLCA STAD HNSC COADREAD}"
EXTRA="${EXTRA:-}"
CONFIGS="${CONFIGS:-configs/blca_hybrid_v2.yaml configs/baselines/blca_survpath.yaml configs/baselines/blca_abmil.yaml configs/baselines/blca_snn.yaml configs/baselines/blca_mlp_omics.yaml}"
PY=python3

run_name() {
  local b; b="$(basename "${1%.yaml}")"; b="${b#blca_}"
  case "$b" in
    hybrid_v2) echo hybrid ;;
    survpath|abmil|snn|mlp_omics) echo "baseline_$b" ;;
    *) echo "$b" ;;
  esac
}

run() {  # run <config> <cancer_lower> <seed>
  local cfg=$1 cancer=$2 seed=$3 name over
  name="$(run_name "$cfg")"; [ "$seed" != "0" ] && name="${name}_seed${seed}"
  over="cancer_type=${cancer},embeddings_dir=${EMB}/${cancer^^},output_dir=${OUT}/${name},num_workers=${WORKERS},seed=${seed}"
  case " $CACHE_COHORTS " in *" ${cancer^^} "*) over="${over},cache_in_ram=true,cache_dtype=float16" ;; esac
  [ -n "$EXTRA" ] && over="${over},${EXTRA}"
  bash scripts/run_experiments.sh "${cfg}::${over}"
}

echo "[queue] $(date '+%F %T') start | cohorts: $COHORTS | seeds: $SEEDS | cache: $CACHE_COHORTS | configs: $CONFIGS | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
for C in $COHORTS; do
  if ! bash scripts/pod/fetch_cohort.sh "$C"; then echo "[queue] $C: fetch failed, skipping"; continue; fi
  for s in $SEEDS; do
    for cfg in $CONFIGS; do run "$cfg" "${C,,}" "$s"; done
  done
done

echo "[queue] $(date '+%F %T') training done; finalizing"
finished=()
for r in "$OUT"/*/*/results.json; do [ -f "$r" ] && finished+=("$(dirname "$r")"); done
[ "${#finished[@]}" -gt 0 ] && $PY scripts/recompute_metrics.py "${finished[@]}"
$PY scripts/aggregate_results.py "$OUT" --out "$OUT/summary_all.md" --compare hybrid
for r in "${finished[@]}"; do $PY scripts/analyze_run.py "$r" >/dev/null 2>&1 || echo "[queue] analysis failed for $r"; done
echo "[queue] $(date '+%F %T') all done -> $OUT/summary_all.md"

if [ "${KEEP_POD:-0}" != "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[queue] removing this pod ($RUNPOD_POD_ID); the volume keeps everything"
  # v2 CLI: `pod remove`; the v1 CLI that https://cli.runpod.net installs: `remove pod`; else the REST API
  runpodctl pod remove "$RUNPOD_POD_ID" 2>/dev/null || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null \
    || curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    || echo "[queue] could not remove the pod automatically - from the laptop: runpodctl pod remove $RUNPOD_POD_ID"
fi
