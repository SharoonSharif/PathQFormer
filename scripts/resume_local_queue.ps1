# Relaunch the local BLCA experiment batch after a reboot or shutdown.
# Everything resumes: finished folds are skipped, an interrupted fold continues from its last epoch.
#
#   powershell -ExecutionPolicy Bypass -File scripts\resume_local_queue.ps1
#
# Starts three detached processes (they survive closing the terminal):
#   1. the main queue (PathQ-Former hybrid, WSI-only, genomics-only, old hyper-parameters)
#   2. the baseline queue, which waits for the main queue to finish
#   3. the finalizer, which waits for the baselines and then recomputes metrics, builds
#      results/summary_all.md and renders figures
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$bash = (Get-Command bash).Source
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

function Launch($args, $log) {
    Start-Process -FilePath $bash -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root "logs\$log.out") -RedirectStandardError (Join-Path $root "logs\$log.err") -PassThru
}

$q1 = Launch 'scripts/run_experiments.sh' 'queue'
$q2 = Launch 'scripts/chain_after.sh logs/queue.out 4 configs/baselines/blca_survpath.yaml configs/baselines/blca_abmil.yaml configs/baselines/blca_snn.yaml configs/baselines/blca_mlp_omics.yaml' 'queue2'
$q3 = Launch 'scripts/finalize_after.sh logs/queue2.out 4 outputs_v2' 'finalize'
"launched: main queue PID $($q1.Id), baseline queue PID $($q2.Id), finalizer PID $($q3.Id)"
"progress: Get-Content logs\queue.out -Wait   |   results: results\summary_all.md when finalize.out says done"
