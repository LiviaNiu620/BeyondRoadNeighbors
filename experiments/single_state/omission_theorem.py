"""Theorem 2 check: the fixed-point error of a mask is pinned by the omitted flux (Table 2)."""
import argparse, json, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator
from brn.experiments.common import *  # noqa
from brn.spectral import entropy_preconditioner
ap = argparse.ArgumentParser(); ap.add_argument("--network", default="Anaheim")
ap.add_argument("--ks", default="4,8,16"); ap.add_argument("--out", default=None)
ap.add_argument("--protocols", default="road,none", help="masks: road (neighbors), road2, road3, ... (kappa-hop), none")
args = ap.parse_args()
from brn.partition import communication_graph
problem, coords, av, logits = load(args.network)
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
Minv = LinearOperator(N.shape, matvec=P.apply, dtype=float)
nN = lambda x: float(np.sqrt(x @ (N @ x)))
nNi = lambda x: float(np.sqrt(x @ P.apply(x)))
out = []
for K in [int(k) for k in args.ks.split(",")]:
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b)
    own = dec.coordinate_owner
    road = set()
    for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
    def masked(pairs):
        At = None; bt = np.zeros_like(b)
        for k, s in enumerate(S):
            m = np.array([(own[i] == k) or ((int(own[i]), k) in pairs) for i in range(own.size)], float)
            R = sparse.diags(m); t = R @ s.matrix @ R
            At = t if At is None else At + t; bt += m * s.rhs
        return At.tocsr(), bt
    protocols = []
    for label in args.protocols.split(","):
        if label == "road": protocols.append((label, road))
        elif label == "none": protocols.append((label, set()))
        else: protocols.append((label, {(s_, r_) for (r_, s_) in communication_graph(dec, label)}))
    for label, pairs in protocols:
        At, bt = masked(pairs)
        lt = spsolve(At.tocsc(), bt)
        phi = (A - At) @ lam - (b - bt)
        lmax_t = eigsh(At, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-8)[0]
        rec = dict(K=K, protocol=label,
                   internal_residual=float(np.linalg.norm(At @ lt - bt) / np.linalg.norm(bt)),
                   err_N=nN(lt - lam) / nN(lam), err_2=float(np.linalg.norm(lt - lam) / np.linalg.norm(lam)),
                   flux=nNi(phi) / nN(lam), lmax_masked=float(lmax_t),
                   dependencies=len(dec.coupling),
                   covered=sum(1 for (r_, s_) in dec.coupling if (s_, r_) in pairs))
        rec["lower"] = rec["flux"] / rec["lmax_masked"]; rec["upper"] = rec["flux"]
        rec["holds"] = bool(rec["lower"] <= rec["err_N"] * (1 + 1e-9) and rec["err_N"] <= rec["upper"] * (1 + 1e-9))
        out.append(rec); print(rec, flush=True)
if args.out: open(args.out, "w").write("\n".join(json.dumps(r) for r in out) + "\n")
