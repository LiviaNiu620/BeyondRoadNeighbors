"""Spectral bounds gamma*, gamma, gamma_reg (Theorem 4) and fixed-round solver accuracy (Table tab:cert)."""
import argparse, json, time
import numpy as np
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator
from brn.experiments.common import *  # noqa
from brn.distributed import DistributedOperator, distributed_cg, CommunicationLedger
from brn.compression import build_compressor
from brn.partition import communication_graph
from brn.spectral import (entropy_preconditioner, congestion_gamma, sharpened_gamma,
                             distributed_spectral_solve)
ap = argparse.ArgumentParser(); ap.add_argument("--network", default="Anaheim")
ap.add_argument("--k", type=int, default=16); ap.add_argument("--theta", type=float, default=0.08)
ap.add_argument("--out", default=None); args = ap.parse_args()
t0 = time.time()
problem, coords, av, logits = load(args.network, theta=args.theta)
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, args.k)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
Minv = LinearOperator(A.shape, matvec=P.apply, dtype=float)
lmax = eigsh(A, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-8)[0]
g1 = congestion_gamma(problem, dec.link_region, hdv, av, args.k)
g2 = sharpened_gamma(problem, dec.link_region, hdv, av, args.k)
rec = dict(network=args.network, K=args.k, theta=args.theta, gamma_exact=float(lmax - 1),
           gamma_cs=g1.gamma_path, gamma_cs_regional=g1.gamma_regional,
           gamma_sharp=g2.gamma_path, gamma_sharp_regional=g2.gamma_regional, rows=[])
print({k: v for k, v in rec.items() if k != "rows"}, flush=True)
ch = communication_graph(dec, "coupling")
def op(c):
    return DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
        touch=dec.coordinate_touch, channels=ch, compressor=build_compressor(c, seed=0),
        ledger=CommunicationLedger())
err = lambda x: float(np.linalg.norm(x - lam) / np.linalg.norm(lam))
for c in ("float64", "uniform8", "uniform6"):
    for R in (1, 2, 3, 4, 6):
        row = dict(compressor=c, rounds=R)
        o = op(c); row["cgN"] = err(distributed_cg(o, b, R, tolerance=0, preconditioner=P.apply).adjoint)
        for nm, up in (("cheb_exact", lmax), ("cheb_sharp", 1 + g2.gamma_path),
                       ("cheb_sharp_reg", 1 + g2.gamma_regional)):
            o = op(c); row[nm] = err(distributed_spectral_solve(o, b, P, up, R).adjoint)
        rec["rows"].append(row); print(row, flush=True)
print("seconds", time.time() - t0)
if args.out: open(args.out, "w").write(json.dumps(rec) + "\n")
