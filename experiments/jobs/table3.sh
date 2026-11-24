#!/usr/bin/env bash
# Table 3 (robustness): EPDA-CG 8 bit R=2 (res_*) and neighbor-only exchange (nb_*) on the common
# ledger, plus the exact reference, across partitions, networks, starts, fire zones and weights.
# Archived records: results/revision/res_*.jsonl, nb_*.jsonl; broadcast column in results/submission/.
source "$(dirname "$0")/_common.sh"
DIR=$OUT_ROOT/table3; mkdir -p "$DIR"
EPDA="N:q8+r2+warm+res"; NB="exactN,roadN@1e-6+warm+res"
# name | environment | run.py options | decision-loop options
CASES=(
  "K4||--rho 2|--network Anaheim --k 4"
  "K8||--rho 2|--network Anaheim --k 8"
  "K32||--rho 2|--network Anaheim --k 32"
  "SF8|RISK_NETWORK=SiouxFalls RISK_K=8|--rho 2|--network SiouxFalls --k 8"
  "seed1||--rho 2|--network Anaheim --k 16 --start perturbed --seed 1"
  "seed2||--rho 2|--network Anaheim --k 16 --start perturbed --seed 2"
  "seed3||--rho 2|--network Anaheim --k 16 --start perturbed --seed 3"
  "seed4||--rho 2|--network Anaheim --k 16 --start perturbed --seed 4"
  "seed5||--rho 2|--network Anaheim --k 16 --start perturbed --seed 5"
  "fire0|RISK_FIRE=0|--rho 2|--network Anaheim --k 16 --theta 0.08"
  "fire6|RISK_FIRE=6|--rho 2|--network Anaheim --k 16 --theta 0.08"
  "fire11|RISK_FIRE=11|--rho 2|--network Anaheim --k 16 --theta 0.08"
  "fire12|RISK_FIRE=12|--rho 2|--network Anaheim --k 16 --theta 0.08"
  "rho1|RISK_FIRE=5|--rho 1|--network Anaheim --k 16 --theta 0.08"
  "rho4|RISK_FIRE=5|--rho 4|--network Anaheim --k 16 --theta 0.08"
)
for c in "${CASES[@]}"; do
  IFS='|' read -r name envs w args <<< "$c"
  job res_$name env $envs $RUN $w $LOOP $args --outer 30 --only "$EPDA" --out "$DIR/res_$name.jsonl"
  job nb_$name  env $envs $RUN $w $LOOP $args --outer 30 --only "$NB"   --out "$DIR/nb_$name.jsonl"
done
