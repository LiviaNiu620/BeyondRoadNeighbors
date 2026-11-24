"""Residual-gap certificate for quantized EPDA-CG: an independent runtime bound (Appendix D.2).

Residual-gap identity:  b - A lam_R = r_R - e_b + sum_k alpha_k e_k   (cold start)
Bound (A >= N):         ||lam_R - lam||_N <= ||r_R||_{N^-1} + ||e_b||_{N^-1} + sum_k |alpha_k| ||e_k||_{N^-1}
Computable version: each e bounded componentwise from the quantization scales that travel
in the message headers (half a step per component), pushed through |A_l| at the regions.
"""
import argparse, json
import numpy as np
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
from brn.spectral import entropy_preconditioner
from brn.distributed import DistributedOperator
from brn.compression import UniformQuantizer
from brn.partition import communication_graph

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
P = entropy_preconditioner(problem, hdv); N = P.matrix()
Ninv = np.linalg.inv(N.toarray() if hasattr(N, "toarray") else np.asarray(N)); absNinv = np.abs(Ninv)
nN = lambda x: float(np.sqrt(x @ (N @ x))); nNi = lambda x: float(np.sqrt(max(x @ (Ninv @ x), 0.0)))
L = 2 ** (args.bits - 1) - 1


class Recording(UniformQuantizer):
    def __call__(self, vector, channel):
        x = np.asarray(vector, dtype=np.float64)
        self.log[channel] = float(np.max(np.abs(x))) if x.size else 0.0
        return super().__call__(vector, channel)


q = Recording(n_bits=args.bits); q.log = {}
channels = communication_graph(dec, "coupling")
op = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
                         touch=dec.coordinate_touch, channels=channels, compressor=q)
absA = [abs(s.matrix) for s in S]


def scale_bound(kind):
    """Componentwise bound on the error of the last exchange, from header scales only.
    Returns (vector-forwarded bound, scalar-per-gather-message bound)."""
    n = op.size; half = lambda ch: q.log.get(ch, 0.0) / (2 * L)
    vec = np.zeros(n); sca = np.zeros(n)
    if kind == "product":
        prod_err = []
        for l in range(op.n_regions):
            d = np.zeros(n)
            for (k, ll), idx in op._index.items():
                if ll == l and k != l:
                    d[idx] = half(f"s{k}->{l}")
            prod_err.append(np.asarray(absA[l] @ d).ravel())
    for (k, l), idx in op._index.items():
        if kind == "product":
            pe = prod_err[l][idx]
            vec[idx] += pe; sca[idx] += pe.max() if pe.size else 0.0
            if k != l:
                vec[idx] += half(f"g{l}->{k}"); sca[idx] += half(f"g{l}->{k}")
        elif k != l:
            vec[idx] += half(f"b{l}->{k}"); sca[idx] += half(f"b{l}->{k}")
    f = lambda e: float(np.sqrt(e @ (absNinv @ e)))
    return f(vec), f(sca)


rec = {"theta": args.theta, "K": args.k, "bits": args.bits, "lam_N": nN(lam)}
for R in [int(r) for r in args.rounds.split(",")]:
    q.log = {}
    bt = op.aggregated_rhs([s.rhs for s in S]); eb = bt - b
    eb_true = nNi(eb); eb_vec, eb_sca = scale_bound("rhs")
    x = np.zeros_like(b); r = bt.copy(); z = P.apply(r); p = z.copy(); rz = float(r @ z)
    s_true = s_vec = s_sca = 0.0; gap = -eb.copy()
    for k in range(R):
        q.log = {}
        qt = op.matvec(p); e = qt - A @ p
        alpha = rz / float(p @ qt)
        ev, es = scale_bound("product")
        s_true += abs(alpha) * nNi(e); s_vec += abs(alpha) * ev; s_sca += abs(alpha) * es
        gap += alpha * e
        x = x + alpha * p; r = r - alpha * qt
        z = P.apply(r); rz_new = float(r @ z); p = z + (rz_new / rz) * p; rz = rz_new
    true_res = b - A @ x
    rN = float(np.sqrt(max(rz, 0.0))); ref = nN(lam)
    rec[f"R{R}"] = dict(
        identity_defect=float(np.linalg.norm(true_res - (r + gap)) / np.linalg.norm(b)),
        true_err=nN(x - lam) / ref, lemma2_rhs=nNi(true_res) / ref,
        bound_true_e=(rN + eb_true + s_true) / ref,
        bound_scales_vector=(rN + eb_vec + s_vec) / ref,
        bound_scales_scalar=(rN + eb_sca + s_sca) / ref,
        r_R=rN / ref, e_b_true=eb_true / ref, sum_true=s_true / ref)
print(json.dumps(rec, indent=1))
if args.out: open(args.out, "w").write(json.dumps(rec, indent=1))
