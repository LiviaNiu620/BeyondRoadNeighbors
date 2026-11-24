"""Single-state accuracy vs bits: state-sharing baselines vs EPDA (same metric)."""
import argparse, json, time
import numpy as np
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
from brn.distributed import DistributedOperator, distributed_cg, CommunicationLedger, local_jacobi_preconditioner
from brn.compression import build_compressor
from brn.partition import communication_graph
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
from brn.sue.operators import link_flows_from_paths
from brn.experiments.state_sharing import Format, StateSystem, protocol_bits, shared_gradient

ap = argparse.ArgumentParser()
ap.add_argument("--network", default="Anaheim"); ap.add_argument("--k", type=int, default=16)
ap.add_argument("--theta", type=float, default=0.08); ap.add_argument("--out", required=True)
ap.add_argument("--formats", default="float64,float32,float16,uniform16,uniform12,uniform10,uniform8,uniform6,log16,log12,log10,log8,log6")
ap.add_argument("--epda", default="all")
args = ap.parse_args()
problem, coords, av, logits = load(args.network, theta=args.theta)
hdv = follower(problem, coords, av, logits)
dec = regions(problem, coords, args.k)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b)
C = sum(s.congestion for s in S); ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
v = link_flows_from_paths(problem, hdv, av); f = np.asarray(hdv, float)
sysm = StateSystem(problem, coords)
lam0, ga0, gu0 = sysm.gradient(v, f, logits)
print("rebuild check", np.linalg.norm(lam0 - lam) / np.linalg.norm(lam), np.linalg.norm(ga0 - ga) / np.linalg.norm(ga))
print("f range", f.min(), np.median(f), f.max(), "v range", v.min(), v.max(), "zeros v", int(np.sum(v == 0)))
rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
def metrics(l, a, u):
    g_hat = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
    g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
    return dict(adjoint_err=rel(l, lam), grad_av_err=rel(a, ga), grad_logit_err=rel(u, gu),
                grad_err=rel(g_hat, g), cosine=float(g_hat @ g / np.linalg.norm(g_hat) / np.linalg.norm(g)))
rows = []
for kind in ("coord", "allgather"):
    for spec in args.formats.split(","):
        fmt = Format(spec)
        l, a, u = shared_gradient(kind, fmt, sysm, dec, v, f, logits)
        rec = dict(protocol=f"{kind}-{spec}", kind=kind, format=spec, **metrics(l, a, u),
                   **protocol_bits(kind, fmt, dec))
        rec["mbit"] = rec["total_bits"] / 1e6
        rows.append(rec); print({k: (round(x, 6) if isinstance(x, float) else x) for k, x in rec.items()}, flush=True)
# EPDA on the same metric and the same per-update ledger as the outer loop

import os as _os
_RESIDUAL_FORM = _os.environ.get("GRAD_FORM", "residual") == "residual"   # revision: AV gradient as N x + r_R (no final product exchange)
NMAT = sum(s.matrix - s.congestion for s in S)
P = entropy_preconditioner(problem, hdv)
jac = local_jacobi_preconditioner([s.matrix for s in S], dec.coordinate_owner)
ch = communication_graph(dec, "coupling")
for label, comp, R, pre in (("EPDA-CG q8 R1", "uniform8", 1, "N"), ("EPDA-CG q8 R2", "uniform8", 2, "N"),
                            ("EPDA-CG q8 R3", "uniform8", 3, "N"), ("EPDA-CG q6 R2", "uniform6", 2, "N"),
                            ("EPDA-CG q6 R1", "uniform6", 1, "N"), ("EPDA-CG f64 R2", "float64", 2, "N"),
                            ("EPDA-CG f64 R4", "float64", 4, "N"),
                            ("Diag-CG q8 R4", "uniform8", 4, "J"), ("Full Diag-CG f64", "float64", 64, "J")):
    if args.epda != "all" and label not in args.epda.split(","):
        continue
    led = CommunicationLedger(); comp_ = build_compressor(comp)
    o = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner, touch=dec.coordinate_touch,
                            channels=ch, compressor=comp_, ledger=led)
    bb = o.aggregated_rhs([s.rhs for s in S])
    sol = distributed_cg(o, bb, R, tolerance=0 if R < 64 else 1e-13,
                         preconditioner=(P.apply if pre == "N" else jac)); x = sol.adjoint
    oc = DistributedOperator(matrices=[s.congestion for s in S], owner=dec.coordinate_owner, touch=dec.coordinate_touch,
                             channels=ch, compressor=comp_, ledger=led)
    a = (NMAT @ x + sol.residual_vector) if _RESIDUAL_FORM else (bb - oc.matvec(x)); u = _logit_gradient(coords, problem, logits, x)
    d = led.as_dict()
    rec = dict(protocol=label, kind="epda", format=comp, rounds=R, **metrics(x, a, u), total_bits=d["total_bits"],
               max_agent_send_bits=d["max_agent_send_bits"], max_agent_receive_bits=d["max_agent_receive_bits"],
               coordinator_bits=0.0)
    rec["mbit"] = rec["total_bits"] / 1e6
    rows.append(rec); print({k: (round(x, 6) if isinstance(x, float) else x) for k, x in rec.items()}, flush=True)
meta = dict(network=args.network, K=args.k, theta=args.theta, n_links=int(v.size), n_paths=int(f.size),
            n_coords=int(lam.size), state_values=int(v.size + f.size))
open(args.out, "w").write("\n".join(json.dumps({**meta, **r}) for r in rows) + "\n")
