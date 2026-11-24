#!/usr/bin/env bash
# Single-state probes behind Figures 1(b)-5, Table 2 and Section 6.3 (initial state, exposure
# objective). Archived records: results/revision/figs/ and results/submission/risk_*.
source "$(dirname "$0")/_common.sh"
DIR=$OUT_ROOT/single_state; mkdir -p "$DIR"
S=experiments/single_state
job risk_trace_Anaheim_K16          env OUT="$DIR/risk_trace_Anaheim_K16.json"         $RUN --rho 2 $S/silent_failure_trace.py   # Fig. 1(b)
job risk_story_checks_Anaheim_K16   env OUT="$DIR/risk_story_checks_Anaheim_K16.json"  $RUN --rho 2 $S/coverage_vs_precision.py  # Fig. 2, Sec. 6.3
for k in 4 8 16 32 64; do                                                                                                       # Fig. 4
  job risk_state_sharing_Anaheim_K$k $RUN --rho 2 $S/protocol_frontier.py --k $k \
      --formats uniform6,uniform8,uniform10,uniform12,uniform16,float64 --out "$DIR/risk_state_sharing_Anaheim_K$k.jsonl"
done
job risk_omission_Anaheim           $RUN --rho 2 $S/omission_theorem.py --ks 8,16 --out "$DIR/risk_omission_Anaheim.jsonl"   # Table 2
job risk_omission_SiouxFalls        env RISK_NETWORK=SiouxFalls RISK_K=8 $RUN --rho 2 $S/omission_theorem.py --network SiouxFalls --ks 8 --out "$DIR/risk_omission_SiouxFalls.jsonl"
job risk_locality_Anaheim           $RUN --rho 2 $S/locality.py --ks 8,16,32,64 --out "$DIR/risk_locality_Anaheim.jsonl"      # Fig. 3(a)
job risk_gamma_share                env OUT="$DIR/risk_gamma_share.json" $RUN --rho 2 $S/congestion_correction.py           # Fig. 3(b)
job risk_fire_share_Anaheim_K16     env BRN_OUTPUTS="$DIR" $RUN --rho 2 $S/fire_zone_flux.py                                  # Sec. 6.3
job risk_fire_sensitivity_Anaheim_K16 env BRN_OUTPUTS="$DIR" $RUN --rho 2 $S/fire_zone_sensitivity.py                         # Sec. 6.3, 6.5
job risk_agents_Anaheim             $RUN --rho 2 $S/per_authority_alignment.py --out "$DIR/risk_agents_Anaheim.jsonl"         # Sec. 6.2
job risk_direction_Anaheim          env OUT="$DIR/risk_direction_Anaheim.jsonl" $RUN --rho 2 $S/gradient_direction.py
job risk_failures_Anaheim_K16       $RUN --rho 2 $S/channel_failures.py --out "$DIR/risk_failures_Anaheim_K16.jsonl"          # Fig. 5
