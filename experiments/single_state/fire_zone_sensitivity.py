"""Fire-zone and exposure-weight sensitivity (single state, Anaheim K=16, nominal demand).

Fire zones: the K=16 regions at originating-demand ranks 1, 4, 8, 12, 16 (rank 1 is the
paper's default).  Weights w_a = 1 + rho exp(-hop_a / ell), ell = 2, rho in {1, 2, 4};
rho = 0 is TSTT.  Only the right-hand side b changes, so A, the dependency graph and the
road mask are shared by every case.

Per case: neighbor-only (road) adjoint error and gradient cosine, and the share of squared
dependency flux supplied by the fire region (contributions k = fire to other owners), over
all dependency pairs and over the pairs that road exchange omits.
Writes results/v2r/risk_fire_sensitivity_Anaheim_K16.json
"""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
from brn.experiments.common import *  # noqa
import brn.assembly as asm
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
from brn.sue.operators import bpr_link_costs, bpr_link_derivatives, link_flows_from_paths, sparse_incidence_matrix
from brn.sue.sparse_sensitivity import sparse_reduced_basis
from brn.experiments import exposure as risk

K = 16
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, K)
own = np.asarray(dec.coordinate_owner)
lreg = np.asarray(dec.link_region)
P = entropy_preconditioner(problem, hdv)
nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0)))
road = set()
for (i, j) in dec.road_adjacency:
    road |= {(int(i), int(j)), (int(j), int(i))}
pairs = sorted({(int(own[i]), int(k)) for i, t in enumerate(dec.coordinate_touch) for k in t if k != own[i]})
omitted = [p for p in pairs if p not in road]

S = asm.assemble_local_systems(problem, coords, dec, hdv, av)
A, _ = centralized_system(S)
A = A.tocsc(); C = sum(s.congestion for s in S)
lu = splu(A)
# road-masked matrix (independent of the objective)
At = None; Ct = None; masks = []
for k, s in enumerate(S):
    heard = {k} | {o for (o, kk) in road if kk == k}
    m = np.isin(own, list(heard)).astype(float); masks.append(m); R = sparse.diags(m)
    t = R @ s.matrix @ R; c = R @ s.congestion @ R
    At = t if At is None else At + t; Ct = c if Ct is None else Ct + c
lut = splu(At.tocsc())
Ak = [s.matrix.tocsr() for s in S]

inc = sparse_incidence_matrix(problem).tocsr(); basis = sparse_reduced_basis(coords).tocsc()
v = link_flows_from_paths(problem, hdv, av)
mc = bpr_link_costs(problem, v) + v * bpr_link_derivatives(problem, v)
links_of = [np.where(lreg == k)[0] for k in range(K)]


def rhs_list(w):
    m = w * mc
    return [np.asarray(basis.T @ (inc[l, :].T @ m[l])).ravel() for l in links_of]


def analyse(bk, fire):
    b = sum(bk); lam = lu.solve(b)
    ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
    g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
    bt = sum(masks[k] * bk[k] for k in range(K)); x = lut.solve(bt)
    a = bt - Ct @ x; u = _logit_gradient(coords, problem, logits, x)
    gh = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
    flux = {}
    for (o, k) in pairs:
        m = (own != o).astype(float); R = sparse.diags(m)
        flux[(o, k)] = nNi((Ak[k] - R @ Ak[k] @ R) @ lam - (1 - m) * bk[k])
    tot = sum(f * f for f in flux.values()); tom = sum(flux[p] ** 2 for p in omitted)
    return dict(road_err=float(np.linalg.norm(x - lam) / np.linalg.norm(lam)),
                road_cos=float(gh @ g / np.linalg.norm(gh) / np.linalg.norm(g)),
                fire_share_all=sum(f * f for (o, k), f in flux.items() if k == fire) / tot,
                fire_share_omitted=sum(flux[(o, k)] ** 2 for (o, k) in omitted if k == fire) / tom,
                omitted_share=tom / tot,
                top68=[list(p) for p in sorted(pairs, key=lambda p: -flux[p])[:68]])


# fire zones by originating demand rank
_, info0 = risk.build_weights(rho=0.0, K=K)
demand = np.zeros(K)
gid = {g.group_id: g for g in problem.groups}; owner = np.asarray(dec.path_owner); seen = set()
for i, p in enumerate(problem.paths):
    if p.group_id in seen:
        continue
    seen.add(p.group_id); g = gid[p.group_id]; demand[owner[i]] += g.hdv_demand + g.av_demand
order = list(np.argsort(-demand))
ranks = [1, 4, 8, 12, 16]
fires = [int(order[r - 1]) for r in ranks]
assert fires[0] == info0["fire_region"], (fires, info0)
deg = {k: len({j for (i, j) in road if i == k}) for k in range(K)}

b_tstt = rhs_list(np.ones(len(problem.links)))
out = dict(K=K, ell=2.0, ranks=ranks, fires=fires, demand=demand.tolist(),
           n_pairs=len(pairs), n_omitted=len(omitted), cases=[])
for r, f in zip(ranks, fires):
    base = analyse(b_tstt, f)
    for rho in (0.0, 1.0, 2.0, 4.0):
        if rho == 0.0:
            res = base
        else:
            w, info = risk.build_weights(rho=rho, ell=2.0, K=K, fire=f)
            res = analyse(rhs_list(w), f)
        top_overlap = len({tuple(p) for p in res["top68"]} & {tuple(p) for p in base["top68"]}) / 68
        rec = dict(fire=f, demand_rank=r, road_degree=deg[f], fire_links=int(len(links_of[f])), rho=rho,
                   overlap_top68_with_tstt=top_overlap, **{k: v for k, v in res.items() if k != "top68"})
        out["cases"].append(rec)
        print(f"rank {r:2d} fire {f:2d} deg {deg[f]} rho {rho:.0f}: road err {rec['road_err']:.3f} cos {rec['road_cos']:.3f} "
              f"fire/all {rec['fire_share_all']:.3f} fire/omitted {rec['fire_share_omitted']:.3f} overlap {top_overlap:.2f}", flush=True)
json.dump(out, open(str(OUTPUTS / "risk_fire_sensitivity_Anaheim_K16.json"), "w"), indent=1)
