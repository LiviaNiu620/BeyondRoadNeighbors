"""Coverage versus precision and what the equilibrium adds (single state, Anaheim; Figure 2, Section 6.3).

(a) What does the equilibrium add?  Split the omitted flux of road-neighbour
    computation into the direct part (b - b~, routes crossing regions with drivers
    held fixed) and the response part ((A - A~) lambda, drivers re-equilibrating),
    and compare the equilibrium-unaware gradient [b; 0] with the true gradient.
(c) Equal-budget coverage vs precision: EPDA-CG (R=2) on flux-ranked channel subsets
    (25/50/75/100% of dependency pairs) at 4/6/8/16/64 bits; error vs bits.
(d) Is the flux ranking computable without the exact adjoint?  Rank pairs with the
    flux evaluated at EPDA's 8-bit, 2-round adjoint and compare.
"""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr
from brn.experiments.common import *  # noqa
from brn.distributed import DistributedOperator, distributed_cg, CommunicationLedger
from brn.compression import build_compressor
from brn.partition import communication_graph
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0))); nN = lambda x: float(np.sqrt(x @ (N @ x)))
out = {}
K = 16
dec = regions(problem, coords, K)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); C = sum(s.congestion for s in S)
own = np.asarray(dec.coordinate_owner)
ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
# (a) equilibrium-unaware gradient: drivers held fixed -> grad_a = b, grad_u = 0
g_dir = np.concatenate([b / np.linalg.norm(ga), 0 * gu])
out["a_cos_unaware_stacked"] = cos(g_dir, g)
out["a_cos_unaware_av"] = cos(b, ga)
out["a_norm_ratio_b_over_Clam"] = float(np.linalg.norm(b) / np.linalg.norm(C @ lam))
road = set()
for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
def masked_parts(pairs):
    At = None; bt = np.zeros_like(b)
    for k, s in enumerate(S):
        heard = {k} | {o for (o, kk) in pairs if kk == k}
        m = np.isin(own, list(heard)).astype(float); R = sparse.diags(m)
        t = R @ s.matrix @ R; At = t if At is None else At + t; bt += m * s.rhs
    return At, bt
At, bt = masked_parts(road)
resp = (A - At) @ lam; direct = -(b - bt); phi = resp + direct
out["a_flux_total"] = nNi(phi) / nN(lam)
out["a_flux_response_part"] = nNi(resp) / nN(lam)
out["a_flux_direct_part"] = nNi(direct) / nN(lam)
lt = spsolve(At.tocsc(), bt)
lt_dir = spsolve(At.tocsc(), b)            # restore only the direct (rhs) part
lt_resp = spsolve(A.tocsc(), bt)           # restore only the response (matrix) part
out["a_err_road"] = float(np.linalg.norm(lt - lam) / np.linalg.norm(lam))
out["a_err_road_if_direct_restored"] = float(np.linalg.norm(lt_dir - lam) / np.linalg.norm(lam))
out["a_err_road_if_response_restored"] = float(np.linalg.norm(lt_resp - lam) / np.linalg.norm(lam))
print({k: round(v, 4) for k, v in out.items()}, flush=True)
# directed pairs and their individual flux, at exact and at approximate adjoint
pairs = sorted({(int(own[i]), int(k)) for i, t in enumerate(dec.coordinate_touch) for k in t if k != own[i]})
Ak = [s.matrix.tocsr() for s in S]
def single_flux(lmb):
    f = {}
    for (o, k) in pairs:
        m = (own != o).astype(float); R = sparse.diags(m)
        f[(o, k)] = nNi((Ak[k] - R @ Ak[k] @ R) @ lmb - (1 - m) * S[k].rhs)
    return f
ch = communication_graph(dec, "coupling")
led = CommunicationLedger()
o_ = DistributedOperator(matrices=[s.matrix for s in S], owner=own, touch=dec.coordinate_touch, channels=ch,
                         compressor=build_compressor("uniform8"), ledger=led)
lam_hat = distributed_cg(o_, o_.aggregated_rhs([s.rhs for s in S]), 2, tolerance=0, preconditioner=P.apply).adjoint
f_ex = single_flux(lam); f_ap = single_flux(lam_hat)
rho = spearmanr([f_ex[p] for p in pairs], [f_ap[p] for p in pairs]).correlation
def err_keep(sel):
    At_, bt_ = masked_parts(sel); x = spsolve(At_.tocsc(), bt_); return float(np.linalg.norm(x - lam) / np.linalg.norm(lam))
for m in (68, 120):
    ex = set(sorted(pairs, key=lambda p: -f_ex[p])[:m]); ap = set(sorted(pairs, key=lambda p: -f_ap[p])[:m])
    out[f"d_err_keep{m}_exact_rank"] = err_keep(ex); out[f"d_err_keep{m}_approx_rank"] = err_keep(ap)
    out[f"d_overlap{m}"] = len(ex & ap) / m
out["d_spearman_flux_exact_vs_approx"] = float(rho)
out["d_approx_adjoint_err"] = float(np.linalg.norm(lam_hat - lam) / np.linalg.norm(lam))
print({k: round(v, 4) for k, v in out.items() if k.startswith("d_")}, flush=True)
# (c) equal-budget frontier: coverage subsets (undirected, ranked by summed directed flux) x precision
und = {}
for (o, k), v in f_ex.items():
    e = (min(o, k), max(o, k)); und[e] = und.get(e, 0.0) + v
ranked = sorted(und, key=lambda e: -und[e])

import os as _os
_RESIDUAL_FORM = _os.environ.get("GRAD_FORM", "residual") == "residual"   # revision: AV gradient as N x + r_R (no final product exchange)
NMAT = sum(s.matrix - s.congestion for s in S)
rows = []
for frac in (0.25, 0.5, 0.75, 1.0):
    keep = frozenset(ranked[:int(round(frac * len(ranked)))])
    for comp in ("uniform4", "uniform6", "uniform8", "uniform16", "float64"):
        led = CommunicationLedger(); cp = build_compressor(comp)
        op = DistributedOperator(matrices=[s.matrix for s in S], owner=own, touch=dec.coordinate_touch, channels=keep, compressor=cp, ledger=led)
        bb = op.aggregated_rhs([s.rhs for s in S])
        sol = distributed_cg(op, bb, 2, tolerance=0, preconditioner=P.apply); x = sol.adjoint
        oc = DistributedOperator(matrices=[s.congestion for s in S], owner=own, touch=dec.coordinate_touch, channels=keep, compressor=cp, ledger=led)
        a = (NMAT @ x + sol.residual_vector) if _RESIDUAL_FORM else (bb - oc.matvec(x)); u = _logit_gradient(coords, problem, logits, x)
        gh = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
        rows.append(dict(coverage=frac, compressor=comp, mbit=led.total_bits / 1e6,
                         grad_err=float(np.linalg.norm(gh - g) / np.linalg.norm(g)), cosine=cos(gh, g)))
        print(rows[-1], flush=True)
out["c_rows"] = rows
json.dump(out, open(__import__("os").environ.get("OUT", str(OUTPUTS / "story_checks_Anaheim_K16.json")), "w"), indent=1)
