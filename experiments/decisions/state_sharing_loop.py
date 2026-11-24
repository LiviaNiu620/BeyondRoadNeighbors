"""Cooperative outer loop with a state-sharing gradient oracle.

Identical outer rule to run_dcsiig (same start, projected AV step, normalized
logit step, clipping, step halving after a TSTT increase, 30 attempts); only
the gradient oracle differs. Bits per attempt are constant (one state exchange).
"""
import argparse, json, time
import numpy as np
from brn.experiments.common import *  # noqa
from brn.records import result_metadata
from brn.sue.operators import bpr_link_costs, link_flows_from_paths, total_system_travel_time
from brn.sue.projection import projected_av_step
from brn.experiments.state_sharing import Format, StateSystem, protocol_bits, shared_gradient

ap = argparse.ArgumentParser()
ap.add_argument("--network", default="Anaheim"); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--outer", type=int, default=30); ap.add_argument("--theta", type=float, default=0.08)
ap.add_argument("--protocols", default="allgather-uniform8,allgather-uniform10,coord-uniform8,coord-uniform10")
ap.add_argument("--out", required=True)
ap.add_argument("--demand-scale", type=float, default=1.0); ap.add_argument("--start", default="uniform"); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
from brn.experiments import demand as scaled; scaled.apply(args.demand_scale)
problem, coords, av0, lg0 = load(args.network, theta=args.theta)
dec = regions(problem, coords, args.k)
if args.start == "perturbed":
    from decision_loop import configure_start
    av0, lg0 = configure_start(problem, coords, av0, lg0, "perturbed", args.seed)
sysm = StateSystem(problem, coords)
solver = LogitEquilibriumSolver(SolverOptions(tolerance=1e-10, max_iterations=200_000, msa_exponent=0.7))

def evaluate(av, lg):
    eq = solver.solve(problem, av_path_flow=av, recommendation=coords.recommendation_from_logits(lg))
    hdv = eq.state.hdv_path_flow
    lf = link_flows_from_paths(problem, hdv, av)
    return hdv, float(total_system_travel_time(lf, bpr_link_costs(problem, lf))), eq.converged

out = open(args.out, "w")
for proto in args.protocols.split(","):
    kind, spec = proto.split("-", 1); fmt = Format(spec)
    per = protocol_bits(kind, fmt, dec)
    t0 = time.perf_counter()
    av, lg = np.array(av0, float), np.array(lg0, float)
    hdv, tstt, ok = evaluate(av, lg); init = tstt
    av_step, logit_step = 1.0, 0.05
    best = (tstt, av.copy(), lg.copy()); trace, bits_trace = [], []; attempts = 0; accepted = 0
    for it in range(args.outer):
        attempts += 1
        v = link_flows_from_paths(problem, hdv, av)
        lam, ga, gu = shared_gradient(kind, fmt, sysm, dec, v, np.asarray(hdv, float), lg)
        step = projected_av_step(coords, av, ga, av_step, 1e-6)
        scale = max(float(np.max(np.abs(gu))), 1e-12)
        cl = np.clip(lg - logit_step * gu / scale, -2.0, 2.0)
        nh, nt, ok = evaluate(step.flow, cl)
        if (not ok) or nt > tstt:
            av_step *= 0.5; logit_step *= 0.5
            if av_step < 1e-10: break
            continue
        av, lg, hdv, tstt = step.flow, cl, nh, nt; accepted += 1
        if tstt < best[0]: best = (tstt, av.copy(), lg.copy())
        trace.append(tstt); bits_trace.append(attempts * per["total_bits"])
    rec = {**result_metadata("cooperative_outer"),
           "config": proto, "kind": kind, "format": spec, "network": args.network, "n_regions": args.k,
           "theta": args.theta, "demand_scale": args.demand_scale, "start_mode": args.start, "start_seed": args.seed, "outer_budget": args.outer, "attempts": attempts, "outer_iterations": accepted,
           "tstt": best[0], "initial_tstt": init, "tstt_improvement": best[0] / init - 1.0,
           "total_bits": attempts * per["total_bits"], "bits_per_attempt": per["total_bits"],
           "max_agent_send_bits": attempts * per["max_agent_send_bits"],
           "max_agent_receive_bits": attempts * per["max_agent_receive_bits"],
           "coordinator_bits": attempts * per["coordinator_bits"],
           "tstt_trace": trace, "bits_trace": bits_trace, "seconds": time.perf_counter() - t0}
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(proto, accepted, f"{rec['tstt_improvement']*100:.3f}%", f"{rec['total_bits']/1e6:.2f} Mbit", f"{rec['seconds']:.0f}s", flush=True)
