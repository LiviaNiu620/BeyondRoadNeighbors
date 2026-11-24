"""Single-state probe: spectral bounds and fixed-round solver accuracy."""
import argparse, json, time
import numpy as np
from scipy.sparse.linalg import spsolve, eigsh
from brn.experiments.common import *  # noqa
from brn.distributed import DistributedOperator, distributed_cg, local_jacobi_preconditioner, CommunicationLedger
from brn.compression import build_compressor
from brn.partition import communication_graph
from brn.spectral import entropy_preconditioner, congestion_gamma, distributed_spectral_solve

ap = argparse.ArgumentParser(); ap.add_argument("--network", default="Anaheim")
ap.add_argument("--k", type=int, default=16); ap.add_argument("--theta", type=float, default=0.08)
ap.add_argument("--rounds", default="1,2,3,4,6,8,12,16"); ap.add_argument("--out", default=None)
args = ap.parse_args()
t0 = time.time()
problem, coords, av, logits = load(args.network, theta=args.theta)
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, args.k)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
C = A - N
from scipy.sparse.linalg import LinearOperator
Minv = LinearOperator(A.shape, matvec=P.apply, dtype=float)
lmax_N = eigsh(A, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-6)[0]
print("lmax", lmax_N, time.time()-t0, flush=True)
# rank(C) <= #links < n, hence lambda_min(N^{-1}A) = 1 exactly
assert problem.number_of_links < A.shape[0]
lmin_N = 1.0
print("lmin", lmin_N, time.time()-t0, flush=True)
jac = local_jacobi_preconditioner([s.matrix for s in S], dec.coordinate_owner)
from scipy import sparse
Dj = sparse.diags(np.sqrt(jac)); Mj = (Dj @ A @ Dj).tocsc()
kj = eigsh(Mj, k=1, which="LA", return_eigenvectors=False)[0] / eigsh(Mj, k=1, sigma=0, which="LM", return_eigenvectors=False)[0]
g = congestion_gamma(problem, dec.link_region, hdv, av, args.k)
rec = {"network": args.network, "K": args.k, "theta": args.theta,
       "kappa_entropy_exact": lmax_N / lmin_N, "lmin_entropy": lmin_N, "lmax_entropy": lmax_N,
       "gamma_path": g.gamma_path, "gamma_regional": g.gamma_regional,
       "kappa_local_jacobi": kj, "rows": []}
print({k: v for k, v in rec.items() if k != "rows"})
channels = communication_graph(dec, "coupling")
def op(comp):
    return DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
                               touch=dec.coordinate_touch, channels=channels,
                               compressor=build_compressor(comp, seed=0), ledger=CommunicationLedger())
for comp in ("float64", "uniform8", "uniform6", "uniform4"):
    for R in [int(r) for r in args.rounds.split(",")]:
        row = {"compressor": comp, "rounds": R}
        o = op(comp); sol = distributed_cg(o, b, max_rounds=R, tolerance=0, preconditioner=jac)
        row["cg_err"] = float(np.linalg.norm(sol.adjoint - lam) / np.linalg.norm(lam)); row["cg_mbit"] = o.ledger.total_bits / 1e6
        for name, up in (("cheb_exact", lmax_N), ("cheb_path", 1 + g.gamma_path), ("cheb_reg", 1 + g.gamma_regional)):
            o = op(comp); s2 = distributed_spectral_solve(o, b, P, up, R, method="chebyshev")
            row[name + "_err"] = float(np.linalg.norm(s2.adjoint - lam) / np.linalg.norm(lam)); row[name + "_mbit"] = o.ledger.total_bits / 1e6
        o = op(comp); s3 = distributed_spectral_solve(o, b, P, 1 + g.gamma_path, R, method="heavyball")
        row["hb_path_err"] = float(np.linalg.norm(s3.adjoint - lam) / np.linalg.norm(lam))
        rec["rows"].append(row)
        print(comp, R, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k.endswith("err") or k.endswith("mbit")})
print("seconds", time.time() - t0)
if args.out:
    open(args.out, "w").write(json.dumps(rec) + "\n")
