#!/usr/bin/env bash
# Pod batch 7 (2026-09-16): sharpen the single-modality baselines and get clean efficiency numbers.
#   1. seeds 1-2 for ABMIL / SNN / MLP under the fixed 20-epoch budget, five cohorts (fast; the seed-0 MLP on
#      RNA was within noise of PathQ-Former on BRCA/COADREAD, so its uncertainty matters for the paper)
#   2. efficiency benchmark on real BLCA patients with the GPU otherwise idle
# Removes its pod at the end. Launch detached (or chain after batch 6 with scripts/pod/chain_on_pod.sh):
#   setsid bash scripts/pod/batch7.sh > /workspace/logs/pod_batch7.out 2>&1 < /dev/null &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CACHE_COHORTS="BLCA STAD HNSC COADREAD BRCA" PYTHONUTF8=1
echo "[batch7] $(date '+%F %T') 1. single-modality baselines, 20 epochs, seeds 1-2"
KEEP_POD=1 OUT=/workspace/outputs_e20 COHORTS="BLCA STAD HNSC COADREAD BRCA" SEEDS="1 2" \
  CONFIGS="configs/protocol_fixed/abmil_e20.yaml configs/protocol_fixed/snn_e20.yaml configs/protocol_fixed/mlp_omics_e20.yaml" bash scripts/pod/run_queue_pod.sh
echo "[batch7] $(date '+%F %T') 2. efficiency (idle GPU)"
python3 scripts/efficiency.py --config configs/protocol_fixed/pathq_fast_e20_aux.yaml --baseline configs/protocol_fixed/survpath_e20.yaml \
  --embeddings_dir /workspace/embeddings/BLCA --n 40 --out /workspace/outputs_e20/efficiency.md 2>&1 | tail -n 8 || echo "[batch7] efficiency failed"
echo "[batch7] $(date '+%F %T') done"
set -a; source /workspace/.env; set +a; [ -f /etc/rp_environment ] && source /etc/rp_environment
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-${RUNPOD_API_KEY_PASTED:-}}"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl pod remove "$RUNPOD_POD_ID" 2>/dev/null || runpodctl remove pod "$RUNPOD_POD_ID" 2>/dev/null \
    || curl -sf -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" || true
fi
