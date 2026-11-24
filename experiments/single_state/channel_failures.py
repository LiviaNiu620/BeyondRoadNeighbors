"""Communication failures (Figure 5): random loss of road-adjacency channels.

The physical channel graph is the road-adjacency graph of the authorities. A fraction
p of its edges fails (20 draws per p). Two delivery rules on the surviving graph:
  relay     : a dependency (o,k) is delivered iff o and k remain connected; cost = hops
  neighbour : delivered iff the direct channel survives
Undelivered dependencies are omitted (masked system, solved to convergence).
Reports coverage, full-system adjoint error, gradient cosine, Theorem-1 bounds and
payload-weighted hop multiplier.
"""
import argparse, json
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator
from brn.experiments.common import *  # noqa
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--draws", type=int, default=20); ap.add_argument("--out", required=True)
args = ap.parse_args()
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
nN = lambda x: float(np.sqrt(x @ (N @ x))); nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0)))
K = args.k
dec = regions(problem, coords, K)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); C = sum(s.congestion for s in S)
ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
own = np.asarray(dec.coordinate_owner)
nok = {}
for i, t in enumerate(dec.coordinate_touch):
    for k in t:
        if k != own[i]: nok[(int(own[i]), int(k))] = nok.get((int(own[i]), int(k)), 0) + 1
dep_und = sorted({(min(o, k), max(o, k)) for (o, k) in nok})
road = sorted(dec.road_adjacency)
def evaluate(pairs_und):
    heard = {k: {k} for k in range(K)}
    for (a_, b_) in pairs_und: heard[a_].add(b_); heard[b_].add(a_)
    At = None; bt = np.zeros_like(b); Ct = None
    for k, s in enumerate(S):
        m = np.isin(own, list(heard[k])).astype(float); R = sparse.diags(m)
        t = R @ s.matrix @ R; c = R @ s.congestion @ R
        At = t if At is None else At + t; Ct = c if Ct is None else Ct + c; bt += m * s.rhs
    x = spsolve(At.tocsc(), bt)
    phi = (A - At) @ lam - (b - bt)
    a = bt - Ct @ x; u = _logit_gradient(coords, problem, logits, x)
    gh = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
    return dict(err_2=float(np.linalg.norm(x - lam) / np.linalg.norm(lam)), err_N=nN(x - lam) / nN(lam),
                flux=nNi(phi) / nN(lam), cosine=float(gh @ g / np.linalg.norm(gh) / np.linalg.norm(g)))
rng = np.random.default_rng(0)
rows = []
for p in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    for d in range(args.draws if p > 0 else 1):
        keep = [e for e in road if rng.random() >= p]
        G = nx.Graph(); G.add_nodes_from(range(K)); G.add_edges_from(keep)
        comps = nx.number_connected_components(G)
        relay = [e for e in dep_und if nx.has_path(G, *e)]
        neigh = [e for e in dep_und if G.has_edge(*e)]
        pay = {e: nok.get(e, 0) + nok.get((e[1], e[0]), 0) for e in dep_und}
        hops = (sum(pay[e] * nx.shortest_path_length(G, *e) for e in relay) / max(sum(pay[e] for e in relay), 1))
        for rule, pairs in (("relay", relay), ("neighbour", neigh)):
            r = evaluate(pairs)
            rows.append(dict(K=K, p=p, draw=d, rule=rule, channels=len(keep), components=comps,
                             coverage=len(pairs) / len(dep_und),
                             payload_coverage=sum(pay[e] for e in pairs) / sum(pay.values()),
                             hop_multiplier=hops if rule == "relay" else 1.0, kept_edges=[list(e) for e in keep], **r))
    sub = [r for r in rows if r["p"] == p]
    for rule in ("relay", "neighbour"):
        q = [r for r in sub if r["rule"] == rule]
        print(f"p={p:.1f} {rule:9s} connected {np.mean([r['components']==1 for r in q]):.2f} cov {np.mean([r['coverage'] for r in q]):.2f} "
              f"err_2 med {np.median([r['err_2'] for r in q]):.3f} cos med {np.median([r['cosine'] for r in q]):.3f} hops {np.mean([r['hop_multiplier'] for r in q]):.2f}", flush=True)
open(args.out, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
