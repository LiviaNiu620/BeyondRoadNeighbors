"""Converged but misdirected: per-round traces of a road-neighbour protocol vs a
coverage-complete protocol (single state, Anaheim K=16).

For R = 1..R_max rounds of entropy-preconditioned CG we record, for each protocol,
the internal residual of the system it actually solves (what the agents can see) and
the error of its adjoint and gradient against the exact ones (what they cannot see).
Run under run_risk.py for the exposure-weighted objective.
"""
import json, os
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
K = int(os.environ.get("K", "16"))
dec = regions(problem, coords, K)
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S); lam = spsolve(A.tocsc(), b); C = sum(s.congestion for s in S)
own = np.asarray(dec.coordinate_owner)
ga = b - C @ lam; gu = _logit_gradient(coords, problem, logits, lam)
g = np.concatenate([ga / np.linalg.norm(ga), gu / np.linalg.norm(gu)])
cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
road = frozenset((min(i, j), max(i, j)) for (i, j) in dec.road_adjacency)
full = communication_graph(dec, "coupling")

import os as _os
_RESIDUAL_FORM = _os.environ.get("GRAD_FORM", "residual") == "residual"   # revision: AV gradient as N x + r_R (no final product exchange)
NMAT = sum(s.matrix - s.congestion for s in S)
rows = []
for name, chans, comp, rmax in (("road", road, "float64", 40), ("complete", full, "uniform8", 8),
                                ("complete64", full, "float64", 8)):
    for R in range(1, rmax + 1):
        led = CommunicationLedger(); cp = build_compressor(comp)
        op = DistributedOperator(matrices=[s.matrix for s in S], owner=own, touch=dec.coordinate_touch,
                                 channels=chans, compressor=cp, ledger=led)
        bb = op.aggregated_rhs([s.rhs for s in S])
        sol = distributed_cg(op, bb, R, tolerance=0, preconditioner=P.apply)
        x = sol.adjoint
        # internal residual of the protocol's own (masked) system, computed exactly
        ex = DistributedOperator(matrices=[s.matrix for s in S], owner=own, touch=dec.coordinate_touch,
                                 channels=chans, compressor=build_compressor("float64"), ledger=CommunicationLedger())
        bt = ex.aggregated_rhs([s.rhs for s in S])
        res = float(np.linalg.norm(bt - ex.matvec(x)) / np.linalg.norm(bt))
        oc = DistributedOperator(matrices=[s.congestion for s in S], owner=own, touch=dec.coordinate_touch,
                                 channels=chans, compressor=cp, ledger=led)
        a = (NMAT @ x + sol.residual_vector) if _RESIDUAL_FORM else (bb - oc.matvec(x)); u = _logit_gradient(coords, problem, logits, x)
        gh = np.concatenate([a / np.linalg.norm(ga), u / np.linalg.norm(gu)])
        rows.append(dict(protocol=name, rounds=R, internal_residual=res,
                         adjoint_err=float(np.linalg.norm(x - lam) / np.linalg.norm(lam)),
                         grad_err=float(np.linalg.norm(gh - g) / np.linalg.norm(g)), cosine=cos(gh, g),
                         mbit=led.total_bits / 1e6))
        print(rows[-1], flush=True)
json.dump(rows, open(os.environ.get("OUT", str(OUTPUTS / "risk_trace_Anaheim_K16.json")), "w"), indent=1)
