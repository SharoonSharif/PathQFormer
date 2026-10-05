#!/usr/bin/env bash
# Pod batch 10 (2026-10-05, reviewer confound check): the official SurvPath trained with PathQ-Former's recipe.
#   1. survpath_recipe_e20       (null tokens + modality dropout 0.15 + aux heads 0.5)   seeds 0-2, five cohorts, DSS, 20 ep
#      survpath_recipe_noaux_e20 (null tokens + modality dropout 0.15, no aux heads)     seeds 0-2, five cohorts, DSS, 20 ep
#      -> /workspace/outputs_e20/<run>[_seed<N>]/<cohort>, same protocol as survpath_e20 / pathq_fast_e20_aux   [30 training runs]
#   2. the post-hoc evaluations every other e20 run received: mean-imputation (``*_impute``) and RNA / WSI
#      permutation (``*_permuted``) conditions via scripts/eval_missing_impute.py --permute (evaluation only, ~2 h)
# Everything is resumable; finished folds / runs are skipped. Removes its pod at the end. Launch detached from the code dir:
#   setsid bash scripts/pod/batch10.sh > /workspace/logs/pod_batch10.out 2>&1 < /dev/null &
# Knobs: STEPS="train evals" (subset to split across pods), SEEDS="0 1 2", KEEP_POD=1 to keep the pod alive.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1
E=/workspace/outputs_e20
COHORTS_LC="blca brca coadread hnsc stad"
RUNS="survpath_recipe_e20 survpath_recipe_noaux_e20"
CONFIGS="configs/protocol_fixed/survpath_recipe_e20.yaml configs/protocol_fixed/survpath_recipe_noaux_e20.yaml"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-train evals}"   # e.g. STEPS="train" on one pod, STEPS="evals" afterwards
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

if has train; then
echo "[batch10] $(date '+%F %T') 1. SurvPath recipe (aux / no-aux), seeds $SEEDS, five cohorts"
KEEP_POD=1 OUT=$E COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="$SEEDS" CONFIGS="$CONFIGS" bash scripts/pod/run_queue_pod.sh
echo "[batch10] $(date '+%F %T') TRAIN_DONE"
fi

if has evals; then
echo "[batch10] $(date '+%F %T') 2. imputation + permutation evals (as batch9 did for survpath_e20 / pathq_fast_e20_aux)"
for s in $SEEDS; do
  sfx=""; [ "$s" != "0" ] && sfx="_seed$s"
  for c in $COHORTS_LC; do
    for r in $RUNS; do
      d=$E/$r$sfx/$c
      [ -f $d/results.json ] || { echo "[batch10] $d: not finished, skipped"; continue; }
      python3 scripts/eval_missing_impute.py $d --permute 2>&1 | tail -n 6
    done
  done
done
python3 scripts/aggregate_results.py "$E" --out "$E/summary_all.md" --compare pathq_fast_e20_aux 2>&1 | tail -n 2
echo "[batch10] $(date '+%F %T') EVALS_DONE"
fi

echo "[batch10] $(date '+%F %T') done"
[ "${KEEP_POD:-0}" = "1" ] && exit 0
set -a; source /workspace/.env; set +a; [ -f /etc/rp_environment ] && source /etc/rp_environment
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-${RUNPOD_API_KEY_PASTED:-}}"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl pod remove "$RUNPOD_POD_ID" 2>/dev/null || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null \
    || curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" || true
fi
