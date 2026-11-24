"""Quantization and route-length checks at the initial state (run through experiments/run.py).

1. Size of the quantization term in Prop. 6(ii) at nu bits (is the SR certificate
   meaningful at 8 bits, or only its contraction term?).
2. AV-gradient error of EPDA-CG at 8 bits, R rounds: congestion form (b - C lam, one more
   scatter-gather) vs residual form (N lam + r_R, no exchange).
3. Regions crossed per reduced coordinate (route length in regions) for K = 4..32.
"""
import argparse, json, math
import numpy as np
from scipy.sparse.linalg import spsolve, eigsh
from brn.experiments.common import *  # noqa
from brn.spectral import entropy_preconditioner
from brn.distributed import DistributedOperator, distributed_cg
from brn.compression import UniformQuantizer
from brn.partition import communication_graph

ap = argparse.ArgumentParser()
ap.add_argument("--network", default="Anaheim"); ap.add_argument("--theta", type=float, default=0.08)
ap.add_argument("--k", type=int, default=16); ap.add_argument("--bits", type=int, default=8)
ap.add_argument("--rounds", default="2,4"); ap.add_argument("--gamma", type=float, required=True,
                help="gamma used by SR (path gamma of Eq. 5)")
ap.add_argument("--out", default=None)
args = ap.parse_args()

problem, coords, av, logits = load(args.network, theta=args.theta)
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
nN = lambda x: float(np.sqrt(x @ (N @ x)))
rec = {"network": args.network, "theta": args.theta, "K": args.k, "bits": args.bits}

# 3. route length in regions
for K in (4, 8, 16, 32):
    d = regions(problem, coords, K)
    t = np.array([len(s) for s in d.coordinate_touch])
    rec[f"regions_per_coord_K{K}"] = dict(mean=float(t.mean()), median=float(np.median(t)),
                                         p90=float(np.percentile(t, 90)), max=int(t.max()))

dec = regions(problem, coords, args.k)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b)
C = sum(s.congestion for s in S)
g_exact = b - C @ lam

# 1. quantization term of Prop. 6(ii)
L = 2 ** (args.bits - 1) - 1; q = 1.0 / (2 * L)
n_k = np.zeros(args.k)
for i, touch in enumerate(dec.coordinate_touch):
    for k in touch: n_k[k] += 1
normA = [float(abs(eigsh(s.matrix.tocsc() if hasattr(s.matrix, "tocsc") else s.matrix, k=1, which="LM",
                          return_eigenvectors=False)[0])) for s in S]
eta = q * sum(math.sqrt(n) * a * (2 + math.sqrt(n) * q) for n, a in zip(n_k, normA))
lmin_N = float(eigsh(N.tocsc(), k=1, sigma=0, which="LM", return_eigenvectors=False)[0])
kap = 1 + args.gamma; rho = (math.sqrt(kap) - 1) / (math.sqrt(kap) + 1)
alpha = 4 / (1 + math.sqrt(kap)) ** 2; Krho = 1 + (1 + 3 * rho) / (1 - rho) ** 2
X = float(np.linalg.norm(lam))                     # iterate bound taken at the solution
quant_term = alpha * Krho * eta * X / math.sqrt(lmin_N) / nN(lam)
rec.update(eta=eta, q=q, n_k_max=int(n_k.max()), normA_max=max(normA), lmin_N=lmin_N,
           rho=rho, quant_term_rel_N=quant_term,
           contraction_R={R: math.sqrt(2) * (1 + 3 * rho) * R * rho ** (R - 1) for R in (5, 9, 21)})

# 2. residual form vs congestion form at nu bits
channels = communication_graph(dec, "coupling")
for R in [int(r) for r in args.rounds.split(",")]:
    comp = UniformQuantizer(n_bits=args.bits)
    op = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
                             touch=dec.coordinate_touch, channels=channels, compressor=comp)
    rhs = b  # exact gather of b, to isolate the step-5 question
    sol = distributed_cg(op, rhs, max_rounds=R, preconditioner=P.apply)
    cop = DistributedOperator(matrices=[s.congestion for s in S], owner=dec.coordinate_owner,
                              touch=dec.coordinate_touch, channels=channels, compressor=comp)
    g_cong = rhs - cop.matvec(sol.adjoint)
    g_res = N @ sol.adjoint + sol.residual_vector
    rel = lambda g: float(np.linalg.norm(g - g_exact) / np.linalg.norm(g_exact))
    rec[f"R{R}"] = dict(adjoint_err=float(np.linalg.norm(sol.adjoint - lam) / np.linalg.norm(lam)),
                        av_grad_err_congestion=rel(g_cong), av_grad_err_residual=rel(g_res))
print(json.dumps(rec, indent=1))
if args.out: open(args.out, "w").write(json.dumps(rec, indent=1))
