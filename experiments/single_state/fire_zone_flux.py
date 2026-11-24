"""Share of the dependency flux that neighbour exchange omits and that the fire zone supplies.

Per directed pair (o,k): flux_ok = || (A_k - R A_k R) lambda - (1-m) b_k ||_{N^-1}, the
contribution region k makes to owner o's coordinates (o's error bound if (o,k) is dropped).
Reports, for TSTT and the exposure objective, the share of squared flux on pairs that road
exchange omits, and among those the share supplied by the fire region.
"""
import json
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
import brn.assembly as asm
from brn.spectral import entropy_preconditioner
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, 16); own = np.asarray(dec.coordinate_owner)
P = entropy_preconditioner(problem, hdv)
nNi = lambda x: float(np.sqrt(max(x @ P.apply(x), 0)))
road = set()
for (i, j) in dec.road_adjacency: road |= {(i, j), (j, i)}
pairs = sorted({(int(own[i]), int(k)) for i, t in enumerate(dec.coordinate_touch) for k in t if k != own[i]})
def flux(S):
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b)
    out = {}
    for (o, k) in pairs:
        m = (own != o).astype(float); R = sparse.diags(m); Ak = S[k].matrix.tocsr()
        out[(o, k)] = nNi((Ak - R @ Ak @ R) @ lam - (1 - m) * S[k].rhs)
    return out
f0 = flux(asm.assemble_local_systems(problem, coords, dec, hdv, av))
from brn.experiments import exposure as risk
info = risk.apply(rho=2.0); fire = info["fire_region"]
f1 = flux(asm.assemble_local_systems(problem, coords, dec, hdv, av))
res = dict(fire_region=fire, n_pairs=len(pairs), n_omitted=sum((o, k) not in road for (o, k) in pairs),
           fire_road_neighbors=sorted({j for (i, j) in road if i == fire}))
for name, f in (("tstt", f0), ("exposure", f1)):
    tot = sum(v * v for v in f.values()); om = {p: v for p, v in f.items() if p not in road}
    tom = sum(v * v for v in om.values())
    res[name] = dict(omitted_share=tom / tot,
                     fire_share_all=sum(v * v for (o, k), v in f.items() if k == fire) / tot,
                     fire_share_of_omitted=sum(v * v for (o, k), v in om.items() if k == fire) / tom,
                     fire_involved_share_of_omitted=sum(v * v for (o, k), v in om.items() if fire in (o, k)) / tom)
print(json.dumps(res, indent=1))
json.dump(res, open(str(OUTPUTS / "risk_fire_share_Anaheim_K16.json"), "w"), indent=1)
