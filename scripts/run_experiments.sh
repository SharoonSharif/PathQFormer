#!/usr/bin/env bash
# Run a queue of PathQ-Former configs sequentially on this machine, logging each to logs/.
# Every run resumes automatically (finished folds are skipped), so re-running the queue is safe.
#
#   bash scripts/run_experiments.sh configs/blca_hybrid_v2.yaml configs/ablation/blca_wsi_only.yaml ...
#   bash scripts/run_experiments.sh            # default queue below
#
# Extra --set overrides for a single config can be given as "config.yaml::key=v,key2=v2".
set -uo pipefail
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
mkdir -p logs

if [ "$#" -eq 0 ]; then
  set -- \
    configs/blca_hybrid_v2.yaml \
    configs/ablation/blca_wsi_only.yaml \
    configs/ablation/blca_genomic_only.yaml \
    configs/blca_baseline_v2.yaml
fi

for item in "$@"; do
  cfg="${item%%::*}"
  overrides=""
  if [[ "$item" == *"::"* ]]; then overrides="${item##*::}"; overrides="${overrides//,/ }"; fi
  name="$(basename "${cfg%.yaml}")"
  # overrides may contain paths; keep the log name flat and short
  [ -n "$overrides" ] && name="${name}_$(echo "$overrides" | tr ' =/' '___' | tr -c 'A-Za-z0-9_.-\n' '_' | cut -c1-120)"
  log="logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] START $cfg $overrides -> $log"
  # shellcheck disable=SC2086
  "$PY" -m src.training.train --config "$cfg" ${overrides:+--set $overrides} > "$log" 2>&1
  status=$?
  echo "[$(date '+%F %T')] END   $cfg (exit $status)"; tail -n 3 "$log"
done

"$PY" scripts/aggregate_results.py outputs_v2 --out results/summary_all.md --compare hybrid >/dev/null 2>&1 && echo "results/summary_all.md updated"
