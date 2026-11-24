"""Why is the omitted contribution mostly direct? Vary driver sensitivity and demand.

For road-neighbour computation, split the omitted flux into direct (b - b~) and
response ((A - A~) lambda) parts, and report the adjoint error when only one part is
restored, together with the exact spectral spread gamma (strength of the response).
"""
import json, sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator
from brn.experiments.common import *  # noqa
import brn.sue.io.tntp as tn
from brn.spectral import entropy_preconditioner
from brn.sue.operators import link_flows_from_paths, bpr_link_costs
_orig = tn.read_tntp_demands
def run(network, K, theta, scale):
    tn.read_tntp_demands = lambda *a, **k: tuple(tn.ODDemandRecord(r.origin, r.destination, r.demand * scale) for r in _orig(*a, **k))
    try:
        problem, coords, av, logits = load(network, theta=theta)
        hdv = follower(problem, coords, av, logits)
    finally:
        tn.read_tntp_demands = _orig
    P = entropy_preconditioner(problem, hdv); N = P.matrix()
    nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0))); nN = lambda x: float(np.sqrt(x @ (N @ x)))
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); own = np.asarray(dec.coordinate_owner)
    C = sum(s.congestion for s in S)
    Minv = LinearOperator(N.shape, matvec=P.apply, dtype=float)
    gam = float(eigsh(A, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-6)[0]) - 1
    road = set()
    for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
    At = None; bt = np.zeros_like(b)
    for k, s in enumerate(S):
        heard = {k} | {o for (o, kk) in road if kk == k}
        m = np.isin(own, list(heard)).astype(float); R = sparse.diags(m)
        t = R @ s.matrix @ R; At = t if At is None else At + t; bt += m * s.rhs
    e = lambda x: float(np.linalg.norm(x - lam) / np.linalg.norm(lam))
    v = link_flows_from_paths(problem, hdv, av); cap = np.array([l.capacity for l in problem.links])
    rec = dict(network=network, K=K, theta=theta, demand_scale=scale, gamma_exact=gam,
               mean_vc=float(np.mean(v / cap)), p90_vc=float(np.percentile(v / cap, 90)),
               flux_direct=nNi(-(b - bt)) / nN(lam), flux_response=nNi((A - At) @ lam) / nN(lam),
               err_road=e(spsolve(At.tocsc(), bt)), err_direct_restored=e(spsolve(At.tocsc(), b)),
               err_response_restored=e(spsolve(A.tocsc(), bt)),
               Clam_over_b=float(np.linalg.norm(C @ lam) / np.linalg.norm(b)))
    print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items()}, flush=True)
    return rec
rows = []
for th in (0.08, 0.32, 1.28):
    rows.append(run("Anaheim", 16, th, 1.0))
for sc in (1.5, 2.0):
    rows.append(run("Anaheim", 16, 0.08, sc))
rows.append(run("Anaheim", 16, 0.32, 2.0))
if not __import__("os").environ.get("ANAHEIM_ONLY"):
    rows.append(run("SiouxFalls", 8, 0.08, 1.0))
json.dump(rows, open(__import__("os").environ.get("OUT", str(OUTPUTS / "response_share.json")), "w"), indent=1)
