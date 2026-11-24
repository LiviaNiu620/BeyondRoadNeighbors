"""How much of the adjoint comes from the drivers' re-equilibration, as load grows?

lambda_0 = N^{-1} b is the adjoint with congestion feedback removed (each OD group
responds to the marginal costs alone).  The response share
    s = ||lambda - lambda_0||_N / ||lambda_0||_N
is bounded by gamma / (1 + gamma) (Proposition, response share).  We vary driver cost
sensitivity theta and demand and report gamma_exact, s, the bound and road-neighbour
flux parts.  Run under run_risk.py for the exposure-weighted objective.
"""
import json, os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator
from brn.experiments.common import *  # noqa
import brn.sue.io.tntp as tn
from brn.spectral import entropy_preconditioner
from brn.sue.operators import link_flows_from_paths
_orig = tn.read_tntp_demands


def run(theta, scale, K=16, network="Anaheim"):
    tn.read_tntp_demands = lambda *a, **k: tuple(tn.ODDemandRecord(r.origin, r.destination, r.demand * scale) for r in _orig(*a, **k))
    try:
        problem, coords, av, logits = load(network, theta=theta)
        hdv = follower(problem, coords, av, logits)
    finally:
        tn.read_tntp_demands = _orig
    P = entropy_preconditioner(problem, hdv); N = P.matrix()
    nN = lambda x: float(np.sqrt(x @ (N @ x))); nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0)))
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); lam0 = P.apply(b)
    Minv = LinearOperator(N.shape, matvec=P.apply, dtype=float)
    gam = float(eigsh(A, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-6)[0]) - 1
    own = np.asarray(dec.coordinate_owner); road = set()
    for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
    At = None; bt = np.zeros_like(b)
    for k, s in enumerate(S):
        heard = {k} | {o for (o, kk) in road if kk == k}
        m = np.isin(own, list(heard)).astype(float); R = sparse.diags(m)
        t = R @ s.matrix @ R; At = t if At is None else At + t; bt += m * s.rhs
    v = link_flows_from_paths(problem, hdv, av); cap = np.array([l.capacity for l in problem.links])
    rec = dict(network=network, K=K, theta=theta, demand_scale=scale, gamma_exact=gam,
               response_share=nN(lam - lam0) / nN(lam0), response_bound=gam / (1 + gam),
               mean_vc=float(np.mean(v / cap)), p90_vc=float(np.percentile(v / cap, 90)),
               flux_direct=nNi(-(b - bt)) / nN(lam), flux_response=nNi((A - At) @ lam) / nN(lam),
               err_road=float(np.linalg.norm(spsolve(At.tocsc(), bt) - lam) / np.linalg.norm(lam)))
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rec.items()}, flush=True)
    return rec


rows = [run(th, sc) for th, sc in ((0.02, 1.0), (0.08, 1.0), (0.32, 1.0), (1.28, 1.0),
                                    (0.08, 1.5), (0.08, 2.0), (0.32, 2.0))]
json.dump(rows, open(os.environ.get("OUT", str(OUTPUTS / "risk_gamma_share.json")), "w"), indent=1)
