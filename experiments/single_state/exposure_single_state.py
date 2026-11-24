"""Single-state checks under the exposure-weighted objective (Anaheim, K=16, nominal demand)."""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
from brn.experiments import common as cm
import brn.assembly as asm
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, 16); own = np.asarray(dec.coordinate_owner)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0))); nN = lambda x: float(np.sqrt(x @ (N @ x)))
road = set()
for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
pairs = sorted({(int(own[i]), int(k)) for i, t in enumerate(dec.coordinate_touch) for k in t if k != own[i]})
def analyse(S):
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); C = sum(s.congestion for s in S)
    ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
    g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
    def masked(sel):
        At = None; bt = np.zeros_like(b); Ct = None
        for k, s in enumerate(S):
            heard = {k} | {o for (o, kk) in sel if kk == k}
            m = np.isin(own, list(heard)).astype(float); R = sparse.diags(m)
            t = R @ s.matrix @ R; c = R @ s.congestion @ R
            At = t if At is None else At + t; Ct = c if Ct is None else Ct + c; bt += m * s.rhs
        return At, bt, Ct
    At, bt, Ct = masked(road); x = spsolve(At.tocsc(), bt)
    a = bt - Ct @ x; u = _logit_gradient(coords, problem, logits, x)
    gh = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
    Ak = [s.matrix.tocsr() for s in S]
    flux = {}
    for (o, k) in pairs:
        m = (own != o).astype(float); R = sparse.diags(m)
        flux[(o, k)] = nNi((Ak[k] - R @ Ak[k] @ R) @ lam - (1 - m) * S[k].rhs)
    return dict(A=A, lam=lam, flux=flux, masked=masked,
                road_err=float(np.linalg.norm(x - lam) / np.linalg.norm(lam)),
                road_cos=float(gh @ g / np.linalg.norm(gh) / np.linalg.norm(g)),
                flux_direct=nNi(-(b - bt)) / nN(lam), flux_resp=nNi((A - At) @ lam) / nN(lam))
S0 = asm.assemble_local_systems(problem, coords, dec, hdv, av)
base = analyse(S0)
from brn.experiments import exposure as risk
info = risk.apply(rho=2.0)
S1 = asm.assemble_local_systems(problem, coords, dec, hdv, av)
wtd = analyse(S1)
out = dict(info=info, A_unchanged=float(abs(base["A"] - wtd["A"]).max()))
for name, r in (("tstt", base), ("exposure", wtd)):
    out[name] = {k: r[k] for k in ("road_err", "road_cos", "flux_direct", "flux_resp")}
rank0 = sorted(pairs, key=lambda p: -base["flux"][p]); rank1 = sorted(pairs, key=lambda p: -wtd["flux"][p])
lam1 = wtd["lam"]
def err_keep(sel):
    At, bt, _ = wtd["masked"](sel); x = spsolve(At.tocsc(), bt); return float(np.linalg.norm(x - lam1) / np.linalg.norm(lam1))
for m in (68, 120):
    out[f"overlap_top{m}"] = len(set(rank0[:m]) & set(rank1[:m])) / m
    out[f"exposure_err_keep{m}_own_rank"] = err_keep(set(rank1[:m]))
    out[f"exposure_err_keep{m}_tstt_rank"] = err_keep(set(rank0[:m]))
fire = info["fire_region"]
for name, r in (("tstt", base), ("exposure", wtd)):
    tot = sum(v ** 2 for v in r["flux"].values())
    out[name]["flux_share_into_fire_region"] = sum(v ** 2 for (o, k), v in r["flux"].items() if k == fire) / tot
print(json.dumps(out, indent=1))
json.dump(out, open(str(OUTPUTS / "risk_single_state_Anaheim_K16.json"), "w"), indent=1)
