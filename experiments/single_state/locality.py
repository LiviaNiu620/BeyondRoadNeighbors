"""Does influence decay with road distance? (Figure 3a)

For each ordered pair (o, k) of authorities at the initial state:
  hops       road-adjacency hop distance d(o, k)
  n_ok       coordinates owned by o that touch region k (EPDA payload per vector)
  flux_ok    ||omitted contribution|| when the (o,k) exchange is dropped (as in pruning)
  infl_eq    ||[grad_a J_k] on o's coordinates||, equilibrium-aware (b_k - C lambda_k)
  infl_dir   ||[b_k] on o's coordinates||, direct effect with drivers held fixed
Each influence is reported as a share of region k's total gradient norm.
"""
import argparse, json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from brn.experiments.common import *  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--network", default="Anaheim")
ap.add_argument("--ks", default="8,16"); ap.add_argument("--out", required=True); args = ap.parse_args()
problem, coords, av, logits = load(args.network)
hdv = follower(problem, coords, av, logits)
rows = []
for K in [int(x) for x in args.ks.split(",")]:
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lu = splu(A.tocsc()); lam = lu.solve(b)
    C = sum(s.congestion for s in S); own = np.asarray(dec.coordinate_owner)
    Bk = np.column_stack([s.rhs for s in S]); Lk = lu.solve(Bk); G = Bk - C @ Lk   # per-region gradients
    adj = {k: set() for k in range(K)}
    for a_, b_ in dec.road_adjacency: adj[a_].add(b_); adj[b_].add(a_)
    dist = np.full((K, K), np.inf)
    for s in range(K):
        dist[s, s] = 0; fr = [s]
        while fr:
            nx = []
            for u in fr:
                for w in adj[u]:
                    if dist[s, w] == np.inf: dist[s, w] = dist[s, u] + 1; nx.append(w)
            fr = nx
    nok = {}
    for i, t in enumerate(dec.coordinate_touch):
        for k in t:
            if k != own[i]: nok[(int(own[i]), int(k))] = nok.get((int(own[i]), int(k)), 0) + 1
    for k, s in enumerate(S):
        Ak = s.matrix.tocsr(); gk = np.linalg.norm(G[:, k]); bkn = np.linalg.norm(Bk[:, k])
        for o in range(K):
            if o == k: continue
            m = (own != o).astype(float); R = sparse.diags(m)
            flux = float(np.linalg.norm((Ak - R @ Ak @ R) @ lam - (1 - m) * s.rhs))
            idx = own == o
            rows.append(dict(K=K, owner=o, region=k, hops=float(dist[o, k]), n_ok=nok.get((o, k), 0),
                             flux=flux / float(np.linalg.norm(b)),
                             infl_eq=float(np.linalg.norm(G[idx, k]) / gk),
                             infl_dir=float(np.linalg.norm(Bk[idx, k]) / bkn)))
    sub = [r for r in rows if r["K"] == K]
    for h in sorted({r["hops"] for r in sub}):
        g = [r for r in sub if r["hops"] == h]
        print(K, "hops", h, "pairs", len(g), "median infl_eq %.3g" % np.median([r["infl_eq"] for r in g]),
              "median infl_dir %.3g" % np.median([r["infl_dir"] for r in g]),
              "frac infl_dir==0 %.2f" % np.mean([r["infl_dir"] == 0 for r in g]),
              "median flux %.3g" % np.median([r["flux"] for r in g]), "median n_ok %d" % np.median([r["n_ok"] for r in g]), flush=True)
open(args.out, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
