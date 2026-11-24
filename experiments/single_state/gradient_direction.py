"""Gradient direction under coverage x precision (single state, Anaheim)."""
import json, sys
import numpy as np
from scipy.sparse.linalg import spsolve
from brn.experiments.common import *  # noqa
from brn.distributed import DistributedOperator, distributed_cg, CommunicationLedger
from brn.compression import build_compressor
from brn.partition import communication_graph
from brn.spectral import entropy_preconditioner
from brn.outer import _logit_gradient
problem, coords, av, logits = load()
hdv = follower(problem, coords, av, logits)
P = entropy_preconditioner(problem, hdv)
out = []
for K in (4, 8, 16, 32):
    dec = regions(problem, coords, K)
    S = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); C = sum(s.congestion for s in S)
    ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
    g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
    for topo, comp, R in (("road", "float64", 64), ("none", "float64", 64), ("coupling", "uniform8", 2), ("coupling", "float64", 64)):
        led = CommunicationLedger(); cp = build_compressor(comp); ch = communication_graph(dec, topo)
        o = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner, touch=dec.coordinate_touch, channels=ch, compressor=cp, ledger=led)
        bb = o.aggregated_rhs([s.rhs for s in S])
        x = distributed_cg(o, bb, R, tolerance=0 if R < 64 else 1e-13, preconditioner=P.apply).adjoint
        oc = DistributedOperator(matrices=[s.congestion for s in S], owner=dec.coordinate_owner, touch=dec.coordinate_touch, channels=ch, compressor=cp, ledger=led)
        a = bb - oc.matvec(x); u = _logit_gradient(coords, problem, logits, x)
        gh = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
        rec = dict(K=K, topology=topo, compressor=comp, rounds=R, cosine=float(gh @ g / np.linalg.norm(gh) / np.linalg.norm(g)),
                   grad_err=float(np.linalg.norm(gh - g) / np.linalg.norm(g)), adjoint_err=float(np.linalg.norm(x - lam) / np.linalg.norm(lam)),
                   mbit=led.total_bits / 1e6)
        out.append(rec); print(rec, flush=True)
open(__import__("os").environ.get("OUT", str(OUTPUTS / "direction_Anaheim.jsonl")), "w").write("\n".join(json.dumps(r) for r in out) + "\n")
