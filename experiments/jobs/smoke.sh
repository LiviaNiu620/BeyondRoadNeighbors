#!/usr/bin/env bash
# Five-minute check that a fresh install reproduces archived records exactly:
# Table 2 (omission theorem, Anaheim K=16) and one Table 3 cell (SiouxFalls, neighbor-only).
source "$(dirname "$0")/_common.sh"
DIR=$OUT_ROOT/smoke; mkdir -p "$DIR"
job omission_K16 $RUN --rho 2 experiments/single_state/omission_theorem.py --ks 16 --out "$DIR/omission_K16.jsonl"
job nb_SF8 env RISK_NETWORK=SiouxFalls RISK_K=8 $RUN --rho 2 $LOOP --network SiouxFalls --k 8 --outer 30 \
    --only "exactN,roadN@1e-6+warm+res" --out "$DIR/nb_SF8.jsonl"
python tools/compare_records.py "$DIR/omission_K16.jsonl" results/submission/risk_omission_Anaheim.jsonl --key K,protocol
python tools/compare_records.py "$DIR/nb_SF8.jsonl" results/revision/nb_SF8.jsonl --key config
