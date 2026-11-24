"""Decision loop under communication failures (Section 6.5): relay vs neighbour-only on chosen draws."""
import argparse, json, time
import numpy as np
from brn.experiments.common import *  # noqa
import brn.outer as outer_mod
from brn import DCSIIGOptions, run_dcsiig
from brn.objective_contract import result_metadata
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--failures", required=True); ap.add_argument("--scenarios", required=True,
               help="comma list p:draw"); ap.add_argument("--outer", type=int, default=30)
ap.add_argument("--out", required=True); args = ap.parse_args()
problem, coords, av0, lg0 = load()
dec = regions(problem, coords, args.k)
F = [json.loads(l) for l in open(args.failures)]
orig = outer_mod.communication_graph
out = open(args.out, "w")
for sc in args.scenarios.split(","):
    p, d = float(sc.split(":")[0]), int(sc.split(":")[1])
    rec_relay = next(r for r in F if r["p"] == p and r["draw"] == d and r["rule"] == "relay")
    keep = [tuple(e) for e in rec_relay["kept_edges"]]
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(args.k)); G.add_edges_from(keep)
    dep = dec.coupling_undirected
    relay = frozenset(e for e in dep if nx.has_path(G, *e))
    neigh = frozenset(e for e in dep if G.has_edge(*e))
    for rule, chans, comp, R, solver, warm, mult in (
            ("relay", relay, "uniform8", 2, "cg_entropy", True, rec_relay["hop_multiplier"]),
            ("neighbour", neigh, "float64", 64, "cg_entropy", False, 1.0)):
        outer_mod.communication_graph = lambda decomposition, topology, _c=chans: _c
        # revision: residual-form AV gradient; forwarding is charged hop by hop on the surviving
        # graph from the per-pair payload record instead of a mean multiplier
        opts = DCSIIGOptions(topology="custom", compressor=comp, cg_rounds=R, solver=solver, warm_start=warm,
                             outer_iterations=args.outer, relay_multiplier=1.0,
                             cg_tolerance=1e-13, gradient_form="residual")
        t0 = time.perf_counter()
        res = run_dcsiig(problem, coords, dec, opts, av0, lg0)
        rec = {**{**result_metadata("cooperative_outer"), "model_version": "v2"}, "scenario": sc, "p": p, "draw": d,
               "rule": rule, "components": rec_relay["components"], "coverage": len(chans) / len(dep),
               "hop_multiplier": mult, "outer_iterations": res.iterations, "tstt": res.tstt,
               "initial_tstt": res.initial_tstt, "tstt_improvement": res.tstt / res.initial_tstt - 1.0,
               **res.ledger.as_dict(), "tstt_trace": res.trace.tstt, "seconds": time.perf_counter() - t0}
        dist = dict(nx.all_pairs_shortest_path_length(G)); led = res.ledger.as_dict()
        hop = 0.0
        for key, bits in led["pair_bits"].items():
            a_, b_ = (int(x) for x in key.split(">"))
            hop += bits * dist[a_][b_]
        rec["hop_charged_bits"] = hop + led["scalar_bits"]
        rec["payload_weighted_hop_multiplier"] = hop / max(led["vector_bits"], 1.0)
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(sc, rule, f"cov {rec['coverage']:.2f}", res.iterations, f"{-rec['tstt_improvement']*100:.3f}%",
              f"{rec['total_bits']/1e6:.1f} Mbit", f"hop-charged {rec['hop_charged_bits']/1e6:.1f} Mbit", f"{rec['seconds']:.0f}s", flush=True)
outer_mod.communication_graph = orig
