"""Check of Lemma 7 (a posteriori certificate) and Corollary 8 (gradient error; Problem 1).

For a coverage-complete protocol that returns lam_hat and forms
    g_M_hat = N lam_hat + r_R   (CG, residual form)   or   b_hat - (C lam_hat)_hat  (step-5 exchange)
    g_u_hat = -F_u^T lam_hat    (local),
the claim is   g_M_hat - g_M = -C e + xi,   g_u_hat - g_u = -F_u^T e,   e = lam_hat - lam,  so
    ||g_M_hat - g_M||_2 <= c_M ||e||_N + ||xi||_2,     c_M = ||C N^{-1/2}||_2 <= gamma* sqrt(lmax(N)),
    ||g_u_hat - g_u||_2 <= c_u ||e||_N,                c_u = ||F_u^T N^{-1/2}||_2.
"""
import argparse, json
import numpy as np
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
from brn.spectral import entropy_preconditioner
from brn.distributed import DistributedOperator
from brn.compression import UniformQuantizer
from brn.partition import communication_graph
from brn.outer import _logit_gradient

ap = argparse.ArgumentParser()
ap.add_argument("--theta", type=float, default=0.08); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--bits", type=int, default=8); ap.add_argument("--rounds", default="2,4")
ap.add_argument("--out", default=None)
args = ap.parse_args()

problem, coords, av, logits = load("Anaheim", theta=args.theta)
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, args.k)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); A = A.tocsr(); lam = spsolve(A.tocsc(), b)
P = entropy_preconditioner(problem, hdv); N = P.matrix(); N = N.toarray() if hasattr(N, "toarray") else np.asarray(N)
C = sum(s.congestion for s in S); C = C.toarray() if hasattr(C, "toarray") else np.asarray(C)
n = len(b)
w, V = np.linalg.eigh(N); Nmh = (V / np.sqrt(w)) @ V.T; Nh = (V * np.sqrt(w)) @ V.T
FuT = -_logit_gradient(coords, problem, logits, np.eye(n))          # F_u^T, shape (paths, n)
nN = lambda x: float(np.sqrt(x @ (N @ x))); Ninv = Nmh @ Nmh
nNi = lambda x: float(np.sqrt(max(x @ (Ninv @ x), 0.0)))
c_M = float(np.linalg.norm(C @ Nmh, 2)); c_u = float(np.linalg.norm(FuT @ Nmh, 2))
c_g = float(np.linalg.norm(np.vstack([C, FuT]) @ Nmh, 2))
gstar = float(np.max(np.linalg.eigvalsh(Nmh @ C @ Nmh))); lmaxN = float(w[-1])
g_M = b - C @ lam; g_u = -FuT @ lam
rec = dict(theta=args.theta, K=args.k, bits=args.bits, n=n, gamma_star=gstar, lmax_N=lmaxN, lmin_N=float(w[0]),
           c_M=c_M, c_M_bound=gstar * np.sqrt(lmaxN), c_u=c_u, c_g=c_g, c_g_bound=float(np.hypot(c_M, c_u)),
           norm_gM=float(np.linalg.norm(g_M)), norm_gu=float(np.linalg.norm(g_u)), lam_N=nN(lam))
# c_u from owner-local blocks only (F_u^T N^{-1/2} is block diagonal by OD group)
owner = np.asarray(dec.coordinate_owner); loc = 0.0
for k in range(dec.n_regions):
    idx = np.where(owner == k)[0]
    if idx.size:
        wk, Vk = np.linalg.eigh(N[np.ix_(idx, idx)])
        loc = max(loc, float(np.linalg.norm(FuT[:, idx] @ ((Vk / np.sqrt(wk)) @ Vk.T), 2)))
rec["c_u_local_max"] = loc
rec["N_offblock_max"] = float(max(np.abs(N[np.ix_(np.where(owner == k)[0], np.where(owner != k)[0])]).max() for k in range(dec.n_regions)))

q = UniformQuantizer(n_bits=args.bits)
channels = communication_graph(dec, "coupling")
op = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
                         touch=dec.coordinate_touch, channels=channels, compressor=q)
opC = DistributedOperator(matrices=[s.congestion for s in S], owner=dec.coordinate_owner,
                          touch=dec.coordinate_touch, channels=channels, compressor=q)
for R in [int(r) for r in args.rounds.split(",")]:
    bt = op.aggregated_rhs([s.rhs for s in S])
    x = np.zeros_like(b); r = bt.copy(); z = P.apply(r); p = z.copy(); rz = float(r @ z)
    for _ in range(R):
        qt = op.matvec(p); alpha = rz / float(p @ qt)
        x = x + alpha * p; r = r - alpha * qt
        z = P.apply(r); rz_new = float(r @ z); p = z + (rz_new / rz) * p; rz = rz_new
    e = x - lam; eta = nN(e); apost = nNi(b - A @ x)
    out = {}
    for form in ("residual", "step5"):
        gM_hat = (N @ x + r) if form == "residual" else (bt - opC.matvec(x))
        xi = gM_hat - g_M + C @ e                       # everything that is not -C e
        eM = float(np.linalg.norm(gM_hat - g_M)); eu = float(np.linalg.norm(-FuT @ x - g_u))
        out[form] = dict(err_M=eM, bound_M=c_M * eta + float(np.linalg.norm(xi)), xi=float(np.linalg.norm(xi)),
                         err_u=eu, bound_u=c_u * eta,
                         rel_M=eM / rec["norm_gM"], rel_u=eu / rec["norm_gu"],
                         rel_bound_M=(c_M * eta + float(np.linalg.norm(xi))) / rec["norm_gM"], rel_bound_u=c_u * eta / rec["norm_gu"],
                         apost_rel_bound_M=(c_M * apost + float(np.linalg.norm(xi))) / rec["norm_gM"],
                         apost_rel_bound_u=c_u * apost / rec["norm_gu"])
    rec[f"R{R}"] = dict(eta_N=eta, eta_rel=eta / rec["lam_N"], apost_N=apost, **out)
print(json.dumps(rec, indent=1))
if args.out: open(args.out, "w").write(json.dumps(rec, indent=1))
