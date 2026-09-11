#!/usr/bin/env bash
# Chain a new batch onto a pod whose running queue would otherwise remove the pod when it finishes.
#
#   bash scripts/pod/chain_on_pod.sh <running-batch-log> <done-marker-regex> <command to launch...>
#   e.g. bash scripts/pod/chain_on_pod.sh /workspace/logs/pod_batch3b.out '\[batch3b\].*done' \
#          bash scripts/pod/batch4.sh 0
#
# How: the running queue's last step calls `runpodctl pod remove <own id>` (then falls back to
# `runpodctl remove pod` and the REST API). We shadow `runpodctl` with a no-op so that first call
# "succeeds" and the fallbacks are never reached; a detached watcher waits for the done marker,
# restores the real runpodctl, and launches the new batch (which removes the pod at ITS end).
# If the watcher dies (pod restart), the pod stays alive: check `runpodctl pod list` from the laptop.
set -uo pipefail
LOG="$1"; MARKER="$2"; shift 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL="$(command -v runpodctl || true)"
if [ -n "$REAL" ] && [ ! -f "${REAL}.real" ]; then
  mv "$REAL" "${REAL}.real"
  printf '#!/bin/sh\nexit 0\n' > "$REAL"; chmod +x "$REAL"
  echo "[chain] shadowed $REAL (real binary at ${REAL}.real)"
fi
CMD="$(printf '%q ' "$@")"   # keeps multi-word arguments such as "1 2" intact
WATCH=/workspace/logs/chain_$(date +%s).sh
cat > "$WATCH" <<EOF
#!/usr/bin/env bash
until grep -qE "$MARKER" "$LOG" 2>/dev/null; do sleep 60; done
echo "[chain] \$(date '+%F %T') marker found in $LOG"
[ -f "${REAL}.real" ] && mv -f "${REAL}.real" "$REAL" && echo "[chain] restored runpodctl"
cd "$REPO" && setsid $CMD > /workspace/logs/pod_chained_\$(date +%H%M%S).out 2>&1 < /dev/null &
echo "[chain] launched: $CMD"
EOF
chmod +x "$WATCH"
setsid bash "$WATCH" > "${WATCH%.sh}.out" 2>&1 < /dev/null &
echo "[chain] watcher $WATCH armed: waits for /$MARKER/ in $LOG, then runs: $*"
