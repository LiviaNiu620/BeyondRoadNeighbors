#!/usr/bin/env bash
# Table 1 and Figure 1(c): decision outcomes, Anaheim, K=16, 30 attempts, three regimes.
# Archived records: results/revision/t1_*.jsonl, h3_*.jsonl.  ~2-3 min per protocol and regime.
source "$(dirname "$0")/_common.sh"
DIR=$OUT_ROOT/table1; mkdir -p "$DIR"
A_NOM="exact,exactN,N:q8+r2+warm,N:q8+r2+warm+res"
A_HIGH="exact,exactN,N:q8+r2+warm+res,N:q8+r4+warm+res,N:q8+r2+warm,N:q8+r4+warm"
B="roadN@1e-6+warm,roadN@1e-6+warm+res,road2N@1e-6+warm,road2N@1e-6+warm+res"
R4="N:q8+r4+warm,N:q8+r4+warm+res"
for g in base theta032 high; do
  case $g in
    base)     W=$NOMINAL;   X=$NOMINAL_ARGS;   A=$A_NOM ;;
    theta032) W=$SENSITIVE; X=$SENSITIVE_ARGS; A=$A_NOM ;;
    high)     W=$HIGH;      X=$HIGH_ARGS;      A=$A_HIGH ;;
  esac
  job t1_${g}_a $RUN $W $LOOP --network Anaheim --k 16 --outer 30 $X --only "$A" --out "$DIR/t1_${g}_a.jsonl"
  job t1_${g}_b $RUN $W $LOOP --network Anaheim --k 16 --outer 30 $X --only "$B" --out "$DIR/t1_${g}_b.jsonl"
  [ $g != high ] && job t1_${g}_r4 $RUN $W $LOOP --network Anaheim --k 16 --outer 30 $X --only "$R4" --out "$DIR/t1_${g}_r4.jsonl"
  job h3_${g} $RUN $W $LOOP --network Anaheim --k 16 --outer 30 $X --only "road3N@1e-6+warm+res" --out "$DIR/h3_${g}.jsonl"
done
job h3_K32 $RUN $NOMINAL $LOOP --network Anaheim --k 32 --outer 30 --only "road2N@1e-6+warm+res,road3N@1e-6+warm+res" --out "$DIR/h3_K32.jsonl"
# state broadcast, coordinator and drivers-held-fixed rows (archived in results/submission/)
SS=experiments/decisions/state_sharing_loop.py; DF=experiments/decisions/drivers_fixed_loop.py
job risk_base_state_Anaheim_K16 $RUN $NOMINAL $SS --k 16 --protocols allgather-uniform8 --out "$DIR/risk_base_state_Anaheim_K16.jsonl"
job risk_theta032_state_Anaheim_K16 $RUN $NOMINAL $SS --k 16 --theta 0.32 --protocols allgather-uniform8 --out "$DIR/risk_theta032_state_Anaheim_K16.jsonl"
job risk_high_state_Anaheim_K16 $RUN $NOMINAL $SS --k 16 --theta 0.32 --demand-scale 2 --protocols allgather-uniform8 --out "$DIR/risk_high_state_Anaheim_K16.jsonl"
job risk_coord_base_Anaheim_K16 $RUN $NOMINAL $SS --k 16 --protocols coord-uniform8 --out "$DIR/risk_coord_base_Anaheim_K16.jsonl"
job risk_coord_theta032_Anaheim_K16 $RUN $NOMINAL $SS --k 16 --theta 0.32 --protocols coord-uniform8 --out "$DIR/risk_coord_theta032_Anaheim_K16.jsonl"
job risk_coord_high_Anaheim_K16 $RUN $NOMINAL $SS --k 16 --theta 0.32 --demand-scale 2 --protocols coord-uniform8 --out "$DIR/risk_coord_high_Anaheim_K16.jsonl"
job risk_unaware_base_Anaheim_K16 $RUN $NOMINAL $DF --theta 0.08 --demand-scale 1.0 --out "$DIR/risk_unaware_base_Anaheim_K16.jsonl"
job risk_unaware_theta032_Anaheim_K16 $RUN $NOMINAL $DF --theta 0.32 --demand-scale 1.0 --out "$DIR/risk_unaware_theta032_Anaheim_K16.jsonl"
job risk_unaware_high_Anaheim_K16 $RUN $NOMINAL $DF --theta 0.32 --demand-scale 2.0 --out "$DIR/risk_unaware_high_Anaheim_K16.jsonl"
