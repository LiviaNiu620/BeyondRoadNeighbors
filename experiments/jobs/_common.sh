# Sourced by the job scripts. Run them from the repository root, e.g.
#   bash experiments/jobs/table1.sh            (outputs/table1/*.jsonl)
#   OUT=/tmp/runs bash experiments/jobs/table1.sh
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
RUN="python -u experiments/run.py"
LOOP=experiments/decisions/decision_loop.py
OUT_ROOT=${OUT:-outputs}
# regimes of the paper (exposure objective rho = 2 throughout)
NOMINAL="--rho 2"                 ; NOMINAL_ARGS="--theta 0.08"
SENSITIVE="--rho 2"               ; SENSITIVE_ARGS="--theta 0.32"
HIGH="--rho 2 --demand-scale 2"   ; HIGH_ARGS="--theta 0.32"
job() {   # job <name> <command...>   (skips finished jobs, logs next to the record)
  local name=$1; shift
  local rec="$DIR/$name.jsonl"
  if [ -s "$rec" ] || [ -s "$DIR/$name.json" ]; then echo "skip $name"; return; fi
  echo "run  $name"
  "$@" > "$DIR/$name.log" 2>&1
}
