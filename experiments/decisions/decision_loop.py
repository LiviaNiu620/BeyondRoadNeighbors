"""v1 cooperative optimization: reference-run gap against a communication budget.

This is the experiment the paper promises. Everything before it measured how
accurately the agents can compute a gradient; this measures what that accuracy is
worth -- the total system travel time actually reached, relative to the same
algorithm run with an exact adjoint, and the total bits spent getting there.

Run:
  python scripts/run_dcsiig.py --network Anaheim --k 16 --outer 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from brn import (  # noqa: E402
    DCSIIGOptions,
    build_partition,
    decompose,
    ensure_upstream_on_path,
    reference_controls,
    run_dcsiig,
    tntp_dir,
)

from brn.records import result_metadata
from brn.partition import road_relay_multiplier
from brn.paths import OUTPUTS

UPSTREAM = ensure_upstream_on_path()

from brn.sue.io.tntp import assemble_tntp_problem  # noqa: E402
from brn.sue.sensitivity import ReducedCoordinates  # noqa: E402

# (label, topology, compressor, cg_rounds, solver, warm_start)
CONFIGURATIONS = (
    ("exact",            "coupling", "float64",   64, "cg_jacobi",  False),
    ("v1:q8+trunc4",     "coupling", "uniform8",   4, "cg_jacobi",  False),
    ("exactN",           "coupling", "float64",   64, "cg_entropy", False),
    ("N:q8+r2",          "coupling", "uniform8",   2, "cg_entropy", False),
    ("N:q8+r2+warm",     "coupling", "uniform8",   2, "cg_entropy", True),
    ("N:q6+r2",          "coupling", "uniform6",   2, "cg_entropy", False),
    ("N:q8+r1",          "coupling", "uniform8",   1, "cg_entropy", False),
    ("N:q6+r1",          "coupling", "uniform6",   1, "cg_entropy", False),
    ("N:q4+r2",          "coupling", "uniform4",   2, "cg_entropy", False),
    ("N:q8+r3+warm",     "coupling", "uniform8",   3, "cg_entropy", True),
    ("N:q8+r4+warm",     "coupling", "uniform8",   4, "cg_entropy", True),
    ("N:q6+r4+warm",     "coupling", "uniform6",   4, "cg_entropy", True),
    ("Rich:q8+r3",       "coupling", "uniform8",   3, "richardson", False),
    ("Rich:q8+r3+warm",  "coupling", "uniform8",   3, "richardson", True),
    ("Rich:q8+r2+warm",  "coupling", "uniform8",   2, "richardson", True),
    ("RichP:q8+r3+warm", "coupling", "uniform8",   3, "richardson_path", True),
    # fair full-precision references: stop at a practical tolerance
    ("exact@1e-4",       "coupling", "float64",   64, "cg_jacobi",  False, 1e-4),
    ("exact@1e-6",       "coupling", "float64",   64, "cg_jacobi",  False, 1e-6),
    ("exactN@1e-4",      "coupling", "float64",   64, "cg_entropy", False, 1e-4),
    ("exactN@1e-6",      "coupling", "float64",   64, "cg_entropy", False, 1e-6),
    ("exactN@1e-4+warm", "coupling", "float64",   64, "cg_entropy", True,  1e-4),
    ("road",             "road",     "float64",   64, "cg_jacobi",  False),
    ("roadN",            "road",     "float64",   64, "cg_entropy", False),
    ("independent",      "none",     "float64",   64, "cg_jacobi",  False),
    # ---- kappa-hop masks, relays, residual form, certified SR ----
    # fixed kappa-hop masks over road adjacency, full precision, converged (as "road")
    ("road2",            "road2",    "float64",   64, "cg_jacobi",  False),
    ("road3",            "road3",    "float64",   64, "cg_jacobi",  False),
    # EPDA's messages relayed over road channels (p = 0): coverage-complete, payload
    # multiplied by the mean road-hop count of the dependency pairs
    ("relay:N:q8+r2+warm", "coupling", "uniform8", 2, "cg_entropy", True),
    ("relay:N:q8+r4+warm", "coupling", "uniform8", 4, "cg_entropy", True),
    # EPDA-CG with the AV gradient formed as N lam + r_R (no step-5 exchange)
    ("N:q8+r2+warm+res", "coupling", "uniform8",   2, "cg_entropy", True),
    ("N:q8+r4+warm+res", "coupling", "uniform8",   4, "cg_entropy", True),
    # certified second-order Richardson at the rounds Prop. 6 asks for
    ("Rich:q8+r4+warm",  "coupling", "uniform8",   4, "richardson", True),
    ("Rich:q8+r6+warm",  "coupling", "uniform8",   6, "richardson", True),
    ("Rich:q8+r8+warm",  "coupling", "uniform8",   8, "richardson", True),
    ("Rich:q8+r13+warm", "coupling", "uniform8",  13, "richardson", True),
    # SR with the computable path gamma of Eq. (5) (one path gather) at the round count
    # Prop. 6(ii) certifies for a 1e-2 contraction: nominal 5, sensitive 9
    ("RichP:q8+r5+warm",  "coupling", "uniform8",  5, "richardson_path", True),
    ("RichP:q8+r9+warm",  "coupling", "uniform8",  9, "richardson_path", True),
)
# ---- common ledger (Tables 1 and 3 of the final paper) ----
CONFIGURATIONS = CONFIGURATIONS + (
    # relay over road channels with the residual-form gradient (no step-5 exchange)
    ("relay:N:q8+r2+warm+res", "coupling", "uniform8", 2, "cg_entropy", True),
    ("relay:N:q8+r4+warm+res", "coupling", "uniform8", 4, "cg_entropy", True),
    # masked protocols on the common ledger: entropy preconditioner, residual 1e-6, warm start
    ("roadN@1e-6+warm",      "road",  "float64", 64, "cg_entropy", True, 1e-6),
    ("roadN@1e-6+warm+res",  "road",  "float64", 64, "cg_entropy", True, 1e-6),
    ("road2N@1e-6+warm",     "road2", "float64", 64, "cg_entropy", True, 1e-6),
    ("road2N@1e-6+warm+res", "road2", "float64", 64, "cg_entropy", True, 1e-6),
    ("road3N@1e-6+warm+res", "road3", "float64", 64, "cg_entropy", True, 1e-6),
)
# high-load certified SR: rounds set from the measured path gamma (see lane_reb3.sh)
for _r in range(10, 41):
    CONFIGURATIONS = CONFIGURATIONS + (
        (f"RichP:q8+r{_r}+warm", "coupling", "uniform8", _r, "richardson_path", True),)
V1_CONFIGURATIONS = (
    ("exact",            "coupling", "float64",   64),
    ("q16",              "coupling", "uniform16", 64),
    ("q8",               "coupling", "uniform8",  64),
    ("q4",               "coupling", "uniform4",  64),
    ("trunc8",           "coupling", "float64",    8),
    ("trunc4",           "coupling", "float64",    4),
    ("trunc2",           "coupling", "float64",    2),
    ("q8+trunc8",        "coupling", "uniform8",   8),
    ("q8+trunc4",        "coupling", "uniform8",   4),
    ("stoch8+trunc4",    "coupling", "stochastic8", 4),
    ("relay8",           "full",     "float64",    8),
    ("road",             "road",     "float64",   64),
    ("road+q8",          "road",     "uniform8",  64),
    ("independent",      "none",     "float64",   64),
)


def road_hop_charged_bits(decomposition, ledger_dict):
    """Payload when every message is forwarded hop by hop over road channels.

    Each directed pair's recorded bits (payload and 64-bit headers) are multiplied by the
    shortest road-path length between the two regions; scalar reductions are added as
    counted. Returns None if some exchanged pair is disconnected in the road graph."""
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(decomposition.n_regions))
    graph.add_edges_from(decomposition.road_adjacency)
    dist = dict(nx.all_pairs_shortest_path_length(graph))
    total = 0.0
    for key, bits in ledger_dict.get("pair_bits", {}).items():
        a, b = (int(x) for x in key.split(">"))
        if b not in dist.get(a, {}):
            return None
        total += bits * dist[a][b]
    return total + float(ledger_dict.get("scalar_bits", 0.0))


def configure_start(problem, coordinates, av_flow, logits, mode: str, seed: int):
    """Return a paired feasible start shared by every selected protocol."""
    if mode == "uniform":
        return av_flow, logits
    if mode != "perturbed":
        raise ValueError(f"unknown start mode {mode!r}")
    rng = np.random.default_rng(seed)
    av = np.array(av_flow, copy=True)
    for group in problem.groups:
        indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
        shares = rng.dirichlet(np.ones(len(indices)))
        av[indices] = group.av_demand * shares
    return av, np.asarray(logits) + 0.2 * rng.standard_normal(logits.size)


def terminal_diagnostics(problem, coordinates, decomposition, av_flow, logits,
                         follower_tolerance=1e-10):
    """Evaluate the returned point against the full centralized system.

    This is an offline diagnostic. It is not counted as information available to
    the agents during the run and is never used to choose an update.
    """
    from brn.sue.equilibrium import LogitEquilibriumSolver, SolverOptions
    from brn.sue.operators import bpr_link_costs, link_flows_from_paths, total_system_travel_time
    from brn.assembly import assemble_local_systems, centralized_system
    from brn.outer import _logit_gradient

    equilibrium = LogitEquilibriumSolver(
        SolverOptions(tolerance=follower_tolerance, max_iterations=200_000,
                      msa_exponent=0.7)
    ).solve(problem, av_path_flow=av_flow,
            recommendation=coordinates.recommendation_from_logits(logits))
    if not equilibrium.converged:
        return {"diagnostic_follower_converged": False}
    hdv = equilibrium.state.hdv_path_flow
    systems = assemble_local_systems(problem, coordinates, decomposition, hdv, av_flow)
    matrix, rhs = centralized_system(systems)
    adjoint = spsolve(matrix.tocsc(), rhs)
    congestion = sum(s.congestion for s in systems)
    av_gradient = rhs - congestion @ adjoint
    logit_gradient = _logit_gradient(coordinates, problem, logits, adjoint)
    from brn.sue.projection import projected_av_step
    mapping = projected_av_step(coordinates, av_flow, av_gradient, 1.0, 1e-6)
    link_flow = link_flows_from_paths(problem, hdv, av_flow)
    tstt = total_system_travel_time(link_flow, bpr_link_costs(problem, link_flow))
    return {
        "diagnostic_follower_converged": True,
        "diagnostic_follower_residual": float(equilibrium.residual),
        "diagnostic_full_adjoint_residual": float(
            np.linalg.norm(matrix @ adjoint - rhs) / max(np.linalg.norm(rhs), 1e-30)
        ),
        "diagnostic_full_av_mapping": float(mapping.normalized_mapping_norm),
        "diagnostic_full_logit_gradient_inf": float(np.max(np.abs(logit_gradient))),
        "diagnostic_tstt": float(tstt),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="Anaheim")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--paths-per-group", type=int, default=3)
    parser.add_argument("--partition", default="spectral")
    parser.add_argument("--outer", type=int, default=40)
    parser.add_argument("--av-step", type=float, default=1.0)
    parser.add_argument("--logit-step", type=float, default=0.05)
    parser.add_argument("--only", default=None, help="comma-separated labels")
    parser.add_argument("--out", default=None)
    parser.add_argument("--budget-mbit", type=float, default=None,
                        help="stop before an outer step once the counted budget is spent")
    parser.add_argument("--start", choices=("uniform", "perturbed"), default="uniform")
    parser.add_argument("--full-diagnostics", action="store_true",
                        help="offline full-system diagnostics after every accepted step")
    parser.add_argument("--av-share", type=float, default=0.25)
    parser.add_argument("--guidance-sensitivity", type=float, default=0.5)
    parser.add_argument("--relay-multiplier", type=float, default=1.0,
                        help="forwarding multiplier for relay emulation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.08)
    parser.add_argument("--line-search", choices=("halve", "armijo"), default="halve",
                        help="halving rule (paper) or per-attempt Armijo backtracking (Appendix D.4)")
    parser.add_argument("--max-backtracks", type=int, default=30)
    args = parser.parse_args()

    data = tntp_dir()
    problem = assemble_tntp_problem(
        data / f"{args.network}_net.tntp",
        data / f"{args.network}_trips.tntp",
        av_share=args.av_share, paths_per_group=args.paths_per_group,
        cost_sensitivity=args.theta, av_cost_sensitivity=args.theta,
        recommendation_sensitivity=args.guidance_sensitivity,
    )
    coordinates = ReducedCoordinates.create(problem)
    av_reduced, logits0 = reference_controls(problem, coordinates)
    av_flow0 = coordinates.flow_from_reduced(av_reduced, "av")
    av_flow0, logits0 = configure_start(
        problem, coordinates, av_flow0, logits0, args.start, args.seed
    )

    node_region = build_partition(problem, args.k, args.partition)
    decomposition = decompose(problem, coordinates, node_region, args.k)
    relay_multiplier = road_relay_multiplier(decomposition)

    print(f"upstream={UPSTREAM}")
    print(f"network={args.network} links={problem.number_of_links} "
          f"paths={problem.number_of_paths} groups={len(problem.groups)} "
          f"coordinates={len(coordinates.nonreference_indices)} K={args.k}")
    print(f"partition: {decomposition.summary()}\n")

    selected = set(args.only.split(",")) if args.only else None
    configurations = [c for c in CONFIGURATIONS
                      if selected is None or c[0] in selected]

    records: list[dict] = []
    reference_tstt = None
    print(f"{'config':14} {'iters':>5} {'TSTT':>14} {'vs exact':>10} "
          f"{'vs start':>9} {'Gbit':>8} {'stat':>9} {'s':>6}")
    for config in configurations:
        label, topology, compressor, rounds, solver, warm = config[:6]
        tolerance = config[6] if len(config) > 6 else 1e-13
        options = DCSIIGOptions(
            cg_tolerance=tolerance,
            topology=topology, compressor=compressor, cg_rounds=rounds,
            solver=solver.replace("_path", ""), warm_start=warm,
            gamma_bound=("sharp_path" if solver.endswith("_path") else "sharp_regional"),
            outer_iterations=args.outer, av_step=args.av_step,
            logit_step=args.logit_step,
            communication_budget_bits=(args.budget_mbit * 1e6
                                       if args.budget_mbit is not None else None),
            seed=args.seed,
            full_diagnostics=args.full_diagnostics,
            relay_multiplier=(relay_multiplier if label.startswith("relay")
                              else args.relay_multiplier),
            gradient_form=("residual" if label.endswith("+res") else "congestion"),
            line_search=args.line_search,
            max_backtracks=args.max_backtracks,
        )
        started = time.perf_counter()
        result = run_dcsiig(problem, coordinates, decomposition, options,
                            av_flow0, logits0)
        elapsed = time.perf_counter() - started
        if label == "exact" or (reference_tstt is None and label == "exactN"):
            reference_tstt = result.tstt
        gap = (result.tstt / reference_tstt - 1.0) if reference_tstt else float("nan")
        improvement = result.tstt / result.initial_tstt - 1.0
        stationarity = result.trace.stationarity[-1] if result.trace.stationarity else float("nan")
        ledger_dict = result.ledger if isinstance(result.ledger, dict) else result.ledger.as_dict()
        hop_charged = road_hop_charged_bits(decomposition, ledger_dict)
        record = {
            **result_metadata("cooperative_outer"),
            "config": label, "topology": topology, "compressor": compressor,
            "cg_rounds": rounds, "network": args.network, "n_regions": args.k,
            "solver": solver, "warm_start": warm, "theta": args.theta,
            "cg_tolerance": tolerance,
            "gamma_hat_trace": result.trace.gamma_hat,
            "outer_iterations": result.iterations,
            "outer_budget": args.outer,
            "tstt": result.tstt, "initial_tstt": result.initial_tstt,
            "gap_vs_exact_reference": gap, "tstt_improvement": improvement,
            "stationarity": stationarity, "seconds": elapsed,
            **result.ledger.as_dict(),
            "road_hop_charged_bits": hop_charged,
            "tstt_trace": result.trace.tstt,
            "bits_trace": result.trace.bits,
            "online_adjoint_residual_trace": result.trace.adjoint_residual,
            "online_stationarity_trace": result.trace.stationarity,
            "full_online_adjoint_residual_trace": result.trace.full_online_adjoint_residual,
            "full_online_av_mapping_trace": result.trace.full_online_av_mapping,
            "full_online_logit_gradient_inf_trace": result.trace.full_online_logit_gradient_inf,
            "start_mode": args.start,
            "start_seed": args.seed,
            "av_share": args.av_share,
            "guidance_sensitivity": args.guidance_sensitivity,
            "relay_multiplier": (relay_multiplier if label.startswith("relay")
                                  else args.relay_multiplier),
            "line_search": args.line_search,
            "max_backtracks": args.max_backtracks,
            "gradient_form": ("residual" if label.endswith("+res") else "congestion"),
            "follower_solves": result.trace.follower_solves,
            "attempt_steps": result.trace.attempt_steps,
            "stalled": result.trace.stalled,
            "accepted_steps": result.iterations,
        }
        record.update(terminal_diagnostics(
            problem, coordinates, decomposition, result.av_flow, result.logits,
        ))
        if args.budget_mbit is not None:
            record["communication_budget_mbit"] = args.budget_mbit
        records.append(record)
        print(f"{label:14} {result.iterations:5d} {result.tstt:14.2f} "
              f"{gap:10.3e} {improvement:9.2%} "
              f"{record['total_bits']/1e9:8.3f} {stationarity:9.2e} {elapsed:6.1f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else (
        OUTPUTS / f"decision_loop_{args.network}_K{args.k}_{stamp}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"\nwrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
