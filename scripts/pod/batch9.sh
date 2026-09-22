#!/usr/bin/env bash
# Pod batch 9 (2026-09-22, reviewer requests). Everything is resumable; finished runs are skipped.
#   1. mean-imputation missing-modality eval for SurvPath and PathQ+aux seeds 1-2 (evaluation only, ~1 h)
#   2. RNA / WSI permutation test for SurvPath and PathQ+aux, seeds 0-2, five cohorts (evaluation only, ~1 h)
#   3. single-modality PathQ-Former seeds 1-2 -> three-seed fusion-vs-ensemble table          [20 training runs]
#   4. SurvPath lr x alpha grid (5 new points) + PathQ+aux lr grid (2 points), seed 0, 5 cohorts [35 training runs]
#   5. late fusion per seed from the single-modality runs
# Removes its pod at the end (three-way chain). Launch detached from the code dir:
#   setsid bash scripts/pod/batch9.sh > /workspace/logs/pod_batch9.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1
E=/workspace/outputs_e20
COHORTS_LC="blca brca coadread hnsc stad"
STEPS="${STEPS:-evals single fusion grid}"   # subset to split the batch across pods, e.g. STEPS="evals single fusion" / STEPS="grid"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

if has evals; then
echo "[batch9] $(date '+%F %T') 1-2. imputation (seeds 1-2) + permutation (seeds 0-2) evals"
for s in "" _seed1 _seed2; do
  for c in $COHORTS_LC; do
    for r in survpath_e20 pathq_fast_e20_aux; do
      d=$E/$r$s/$c
      [ -f $d/results.json ] || continue
      if [ -n "$s" ]; then
        python3 scripts/eval_missing_impute.py $d --permute 2>&1 | tail -n 6
      else
        python3 scripts/eval_missing_impute.py $d --permute --no-impute 2>&1 | tail -n 6
      fi
    done
  done
done
echo "[batch9] $(date '+%F %T') EVALS_DONE"
fi

if has single; then
echo "[batch9] $(date '+%F %T') 3. single-modality PathQ-Former, seeds 1-2"
KEEP_POD=1 OUT=$E COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" \
  CONFIGS="configs/protocol_fixed/pathq_e20_wsi_only.yaml configs/protocol_fixed/pathq_e20_genomic_only.yaml" bash scripts/pod/run_queue_pod.sh
fi

if has fusion; then
echo "[batch9] $(date '+%F %T') 5. late fusion per seed"
for s in "" _seed1 _seed2; do
  for c in $COHORTS_LC; do
    a=$E/pathq_e20_wsi_only$s/$c; b=$E/pathq_e20_genomic_only$s/$c
    [ -f $a/results.json ] && [ -f $b/results.json ] && python3 scripts/late_fusion.py $a $b --out $E/late_fusion_e20$s/$c 2>&1 | tail -n 2
  done
done
fi

if has grid; then
echo "[batch9] $(date '+%F %T') 4. hyper-parameter grid (SurvPath lr x alpha, PathQ+aux lr), seed 0"
KEEP_POD=1 OUT=/workspace/outputs_grid COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="0" \
  CONFIGS="$(ls configs/grid/*.yaml | tr '\n' ' ')" bash scripts/pod/run_queue_pod.sh
fi

echo "[batch9] $(date '+%F %T') done"
set -a; source /workspace/.env; set +a; [ -f /etc/rp_environment ] && source /etc/rp_environment
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-${RUNPOD_API_KEY_PASTED:-}}"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl pod remove "$RUNPOD_POD_ID" 2>/dev/null || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null \
    || curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" || true
fi
