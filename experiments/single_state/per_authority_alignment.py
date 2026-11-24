"""Agent-level view: who loses under road exchange, and who carries the load.

Per authority k at the initial state:
  cos_road / cos_epda : cosine between k's own gradient block (AV + logit coordinates
                        k owns) under the protocol and the exact block
  exposure            : flow-weighted share of the links on k's owned paths that lie
                        outside k and its road neighbours
Per-agent send/receive bits per gradient for EPDA (8b R2), all-gather and coordinator
(10-bit state, comparable gradient error).
"""
import argparse, json
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr
from brn.experiments.common import *  # noqa
from brn.distributed import DistributedOperator, distributed_cg, CommunicationLedger
from brn.compression import build_compressor
from brn.partition import communication_graph
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
from brn.experiments.state_sharing import Format, state_messages
ap = argparse.ArgumentParser(); ap.add_argument("--ks", default="16,32"); ap.add_argument("--out", required=True)
args = ap.parse_args()
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv)
flow = np.asarray(hdv) + np.asarray(av)
rows = []
for K in [int(x) for x in args.ks.split(",")]:
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); C = sum(s.congestion for s in S)
    ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
    own = np.asarray(dec.coordinate_owner); lreg = np.asarray(dec.link_region); powner = np.asarray(dec.path_owner)
    nb = {k: {k} for k in range(K)}
    for a_, b_ in dec.road_adjacency: nb[a_].add(b_); nb[b_].add(a_)
    def solve(topo, comp, R):
        led = CommunicationLedger(); cp = build_compressor(comp); ch = communication_graph(dec, topo)
        o = DistributedOperator(matrices=[s.matrix for s in S], owner=own, touch=dec.coordinate_touch, channels=ch, compressor=cp, ledger=led)
        bb = o.aggregated_rhs([s.rhs for s in S])
        x = distributed_cg(o, bb, R, tolerance=0 if R < 64 else 1e-13, preconditioner=P.apply).adjoint
        oc = DistributedOperator(matrices=[s.congestion for s in S], owner=own, touch=dec.coordinate_touch, channels=ch, compressor=cp, ledger=led)
        return bb - oc.matvec(x), _logit_gradient(coords, problem, logits, x), led
    ra, ru, _ = solve("road", "float64", 64)
    ea, eu, eled = solve("coupling", "uniform8", 2)
    msgs = state_messages(dec); f10 = Format("uniform10")
    sb = [f10.bits(l.size) + f10.bits(p.size) for l, p in msgs]
    def cos(xa, xu, idx):
        v = np.concatenate([xa[idx] / np.linalg.norm(ga), xu[idx] / np.linalg.norm(gu)])
        w = np.concatenate([ga[idx] / np.linalg.norm(ga), gu[idx] / np.linalg.norm(gu)])
        return float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)))
    for k in range(K):
        idx = own == k
        if not idx.any():
            continue
        paths = np.where(powner == k)[0]
        out_len = tot_len = 0.0
        for p in paths:
            li = np.asarray(problem.paths[p].link_indices)
            outside = ~np.isin(lreg[li], list(nb[k]))
            out_len += flow[p] * outside.sum(); tot_len += flow[p] * li.size
        nk = int(idx.sum())
        rows.append(dict(K=K, agent=k, coords=nk, exposure=out_len / tot_len,
                         cos_road=cos(ra, ru, idx), cos_epda=cos(ea, eu, idx),
                         err_road=float(np.linalg.norm(np.concatenate([ra[idx]-ga[idx], ru[idx]-gu[idx]])) / np.linalg.norm(np.concatenate([ga[idx], gu[idx]]))),
                         epda_send=eled.per_agent_send_bits.get(k, 0.0), epda_recv=eled.per_agent_receive_bits.get(k, 0.0),
                         ag_send=(K - 1) * sb[k], ag_recv=sum(sb) - sb[k],
                         co_send=sb[k] + 0.0, co_recv=2 * f10.bits(nk)))
    sub = [r for r in rows if r["K"] == K]
    rho = spearmanr([r["exposure"] for r in sub], [r["cos_road"] for r in sub])
    print(K, "spearman(exposure, cos_road) = %.2f p=%.3g" % (rho.correlation, rho.pvalue))
    for key in ("epda", "ag", "co"):
        s = np.array([r[f"{key}_send"] + r[f"{key}_recv"] for r in sub]) / 1e3
        print(f"   {key} per-agent kbit/gradient: mean {s.mean():.1f} max {s.max():.1f} max/mean {s.max()/s.mean():.2f}")
    print("   cos_road min/median %.2f/%.2f  cos_epda min %.5f" % (min(r["cos_road"] for r in sub), np.median([r["cos_road"] for r in sub]), min(r["cos_epda"] for r in sub)))
open(args.out, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
