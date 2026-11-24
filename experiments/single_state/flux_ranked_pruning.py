"""Selective omission revisited: rank dependency edges with Theorem 2 itself.

For a kept set S of directed (owner o -> region k) exchanges, region k hears the
owners {k} U {o : (o,k) in S}, and the omitted flux is
    phi(S) = sum_k (A_k - R_k A_k R_k) lambda - (1 - m_k) * b_k,
which is NOT additive over pairs (cross blocks between two unheard owners).
Rules compared at equal edge budgets:
  road     : road-adjacent pairs (both directions)
  pairnorm : rank pairs by their individual flux norm (previous version)
  random   : mean over 5 random subsets
  greedy   : forward greedy on ||phi(S)||_{N^{-1}}, the upper bound of Theorem 2
Reported error = full-system adjoint error of the masked fixed point.
"""
import argparse, json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
from brn.spectral import entropy_preconditioner
ap = argparse.ArgumentParser(); ap.add_argument("--network", default="Anaheim")
ap.add_argument("--ks", default="8,16"); ap.add_argument("--out", required=True); args = ap.parse_args()
problem, coords, av, logits = load(args.network)
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0.0)))
nN = lambda x: float(np.sqrt(x @ (N @ x)))
out = []
for K in [int(x) for x in args.ks.split(",")]:
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); own = np.asarray(dec.coordinate_owner)
    Ak = [s.matrix.tocsr() for s in S]
    pairs = sorted({(int(own[i]), int(k)) for i, t in enumerate(dec.coordinate_touch) for k in t if k != own[i]})
    def phi_region(k, heard):
        m = np.isin(own, list(heard)).astype(float); R = sparse.diags(m)
        return (Ak[k] - R @ Ak[k] @ R) @ lam - (1 - m) * S[k].rhs
    def masked(kept):
        At = None; bt = np.zeros_like(b)
        for k, s in enumerate(S):
            heard = {k} | {o for (o, kk) in kept if kk == k}
            m = np.isin(own, list(heard)).astype(float); R = sparse.diags(m)
            t = R @ s.matrix @ R; At = t if At is None else At + t; bt += m * s.rhs
        x = spsolve(At.tocsc(), bt)
        return dict(err_2=float(np.linalg.norm(x - lam) / np.linalg.norm(lam)), err_N=nN(x - lam) / nN(lam))
    def flux_of(kept):
        tot = np.zeros_like(b)
        for k in range(K):
            tot += phi_region(k, {k} | {o for (o, kk) in kept if kk == k})
        return nNi(tot) / nN(lam)
    road = set()
    for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
    road &= set(pairs)
    single = {p: nNi(phi_region(p[1], {p[1]} | {o for o in range(K) if o != p[0]})) for p in pairs}
    ranked = sorted(pairs, key=lambda p: -single[p])
    budgets = sorted({len(road), len(pairs) // 4, len(pairs) // 2, 3 * len(pairs) // 4})
    # greedy forward selection on the Theorem-1 bound
    region_phi = [phi_region(k, {k}) for k in range(K)]; heard = [{k} for k in range(K)]
    total = sum(region_phi); kept = []; greedy_at = {}
    remaining = set(pairs)
    while len(kept) < max(budgets):
        best = None
        for (o, k) in remaining:
            new_k = phi_region(k, heard[k] | {o})
            val = nNi(total - region_phi[k] + new_k)
            if best is None or val < best[0]: best = (val, (o, k), new_k)
        val, (o, k), new_k = best
        total = total - region_phi[k] + new_k; region_phi[k] = new_k; heard[k].add(o)
        kept.append((o, k)); remaining.discard((o, k))
        if len(kept) in budgets: greedy_at[len(kept)] = list(kept)
    rng = np.random.default_rng(0)
    for m in budgets:
        rec = dict(K=K, pairs=len(pairs), budget=m, road_pairs=len(road))
        if m == len(road):
            r = masked(road); rec.update(road_err_2=r["err_2"], road_err_N=r["err_N"], road_flux=flux_of(road))
        r = masked(set(ranked[:m])); rec.update(pairnorm_err_2=r["err_2"], pairnorm_err_N=r["err_N"], pairnorm_flux=flux_of(set(ranked[:m])))
        g = set(greedy_at[m]); r = masked(g); rec.update(greedy_err_2=r["err_2"], greedy_err_N=r["err_N"], greedy_flux=flux_of(g))
        rs = []
        for _ in range(5):
            sel = set(map(tuple, np.array(pairs)[rng.choice(len(pairs), m, replace=False)].tolist()))
            rs.append(masked(sel)["err_2"])
        rec.update(random_err_2_mean=float(np.mean(rs)), random_err_2_min=float(np.min(rs)))
        out.append(rec); print(rec, flush=True)
open(args.out, "w").write("\n".join(json.dumps(r) for r in out) + "\n")
