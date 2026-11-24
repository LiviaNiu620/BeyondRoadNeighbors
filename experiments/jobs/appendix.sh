#!/usr/bin/env bash
# Appendix D: kappa-hop masks, certificates, line search, second-order Richardson, channel-failure
# decision loop. Archived records: results/extended/ and results/revision/.
source "$(dirname "$0")/_common.sh"
DIR=$OUT_ROOT/appendix; mkdir -p "$DIR"
S=experiments/single_state; C=experiments/certificates; D=experiments/decisions
# D.3  kappa-hop masks: coverage and Theorem 2 interval (decision-loop rows are in table1.sh)
job omission_khop_Anaheim $RUN --rho 2 $S/omission_theorem.py --ks 8,16 --protocols road,road2,road3,none --out "$DIR/omission_khop_Anaheim.jsonl"
job omission_khop_K32     $RUN --rho 2 $S/omission_theorem.py --ks 32 --protocols road,road2,road3,road4 --out "$DIR/omission_khop_K32.jsonl"
# D.2  spectral bounds, certified rounds, a posteriori certificate and gradient bound
job bounds_base     $RUN $NOMINAL   $C/spectral_bounds.py --theta 0.08 --out "$DIR/bounds_base.json"
job bounds_theta032 $RUN $SENSITIVE $C/spectral_bounds.py --theta 0.32 --out "$DIR/bounds_theta032.json"
job bounds_high     $RUN $HIGH      $C/spectral_bounds.py --theta 0.32 --out "$DIR/bounds_high.json"
for g in base theta032 high; do
  case $g in base) W=$NOMINAL; X=$NOMINAL_ARGS ;; theta032) W=$SENSITIVE; X=$SENSITIVE_ARGS ;; high) W=$HIGH; X=$HIGH_ARGS ;; esac
  job gradient_corollary_$g $RUN $W $C/gradient_certificate.py $X --out "$DIR/gradient_corollary_$g.json"
  job runtime_bound_$g      $RUN $W $C/residual_gap_certificate.py $X --out "$DIR/runtime_bound_$g.json"
done
# D.4  Armijo line search
job r_base_armijo_exact     $RUN $NOMINAL   $LOOP --network Anaheim --k 16 --outer 30 --line-search armijo --only exact --out "$DIR/r_base_armijo_exact.jsonl"
job r_base_armijo_road      $RUN $NOMINAL   $LOOP --network Anaheim --k 16 --outer 30 --line-search armijo --only road --out "$DIR/r_base_armijo_road.jsonl"
job r_theta032_armijo_road  $RUN $SENSITIVE $LOOP --network Anaheim --k 16 --outer 30 --theta 0.32 --line-search armijo --only road --out "$DIR/r_theta032_armijo_road.jsonl"
job r_high_armijo_road      $RUN $HIGH      $LOOP --network Anaheim --k 16 --outer 30 --theta 0.32 --line-search armijo --only road --out "$DIR/r_high_armijo_road.jsonl"
# D.5  second-order Richardson at the certified round counts (path gamma; 1e-2 contraction)
job r_base_richP_r5      $RUN $NOMINAL   $LOOP --network Anaheim --k 16 --outer 30 --only "RichP:q8+r5+warm" --out "$DIR/r_base_richP_r5.jsonl"
job r_theta032_richP_r9  $RUN $SENSITIVE $LOOP --network Anaheim --k 16 --outer 30 --theta 0.32 --only "RichP:q8+r9+warm" --out "$DIR/r_theta032_richP_r9.jsonl"
job r_high_richP_r21     $RUN $HIGH      $LOOP --network Anaheim --k 16 --outer 30 --theta 0.32 --only "RichP:q8+r21+warm" --out "$DIR/r_high_richP_r21.jsonl"
# Sec. 6.5  decision loop under channel failures (p = 0.3; connected and partitioned draw)
job risk_failures_outer_Anaheim_K16 $RUN --rho 2 $D/channel_failure_loop.py --failures results/submission/failures_Anaheim_K16.jsonl \
    --scenarios 0.3:0,0.3:1 --out "$DIR/risk_failures_outer_Anaheim_K16.jsonl"
