"""Quick start: the silent failure of neighbor-only exchange and its fix, on one equilibrium state.

    python examples/quickstart.py [--network SiouxFalls] [--k 8]

Builds the instance, splits the adjoint system A lam = b into regional contributions, compares
the road-neighbor mask with full coverage of the dependency graph G_dep, and runs EPDA
(entropy-preconditioned distributed CG, 8-bit messages, two rounds). Takes seconds on SiouxFalls.
"""
import argparse

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from brn import assemble_local_systems, centralized_system
from brn.compression import build_compressor
from brn.distributed import CommunicationLedger, DistributedOperator, distributed_cg
from brn.experiments.common import follower, load, regions
from brn.partition import communication_graph
from brn.spectral import entropy_preconditioner, sharpened_gamma

ap = argparse.ArgumentParser()
ap.add_argument("--network", default="SiouxFalls")
ap.add_argument("--k", type=int, default=8)
args = ap.parse_args()

# 1. Instance: TNTP network, three candidate paths per OD group, logit drivers, 25% AVs.
problem, coords, av, logits = load(args.network)
hdv = follower(problem, coords, av, logits)            # driver equilibrium for these controls
dec = regions(problem, coords, args.k)                 # K authorities by spectral bisection

# 2. Regional decomposition (Proposition 1): A = sum_k A_k, b = sum_k b_k.
S = assemble_local_systems(problem, coords, dec, hdv, av)
A, b = centralized_system(S)
lam = spsolve(A.tocsc(), b)
P = entropy_preconditioner(problem, hdv)               # owner-held entropy blocks N
road = {(i, j) for e in dec.road_adjacency for (i, j) in (e, e[::-1])}
deps = [(r, s_) for (r, s_) in dec.coupling if r != s_]
print(f"{args.network}, K={args.k}: {len(road)} directed road channels, "
      f"{len(deps)} directed dependencies in G_dep")


# 3. A fixed mask solves a different system (Eq. 3); its error is pinned by the omitted flux.
def masked_solve(pairs):
    own = dec.coordinate_owner
    At, bt = None, np.zeros_like(b)
    for k, s in enumerate(S):
        keep = np.array([own[i] == k or (int(own[i]), k) in pairs for i in range(own.size)], float)
        R = sparse.diags(keep)
        At = R @ s.matrix @ R if At is None else At + R @ s.matrix @ R
        bt += keep * s.rhs
    return At.tocsr(), bt


n_norm = lambda x: float(np.sqrt(x @ (P.matrix() @ x)))
At, bt = masked_solve(road)
lt = spsolve(At.tocsc(), bt)
phi = (A - At) @ lam - (b - bt)
print(f"neighbor-only: own residual {np.linalg.norm(At @ lt - bt) / np.linalg.norm(bt):.1e}, "
      f"true adjoint error {n_norm(lt - lam) / n_norm(lam):.3f} (N-norm), "
      f"omitted flux {np.sqrt(phi @ P.apply(phi)) / n_norm(lam):.3f}")

# 4. EPDA: cover G_dep, precondition with N, send 8-bit messages; rounds follow gamma (Thm. 4).
gamma = sharpened_gamma(problem, dec.link_region, hdv, av, args.k).gamma_path
print(f"spectral bound: spec(N^-1 A) in [1, {1 + gamma:.2f}]")
for rounds in (2, 4, 8):
    op = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
                             touch=dec.coordinate_touch, channels=communication_graph(dec, "coupling"),
                             compressor=build_compressor("uniform8", seed=0), ledger=CommunicationLedger())
    sol = distributed_cg(op, b, rounds, tolerance=0, preconditioner=P.apply)
    r = b - A @ sol.adjoint
    print(f"EPDA, 8 bit, {rounds} rounds: adjoint error {n_norm(sol.adjoint - lam) / n_norm(lam):.4f} "
          f"(a posteriori certificate {np.sqrt(r @ P.apply(r)) / n_norm(lam):.4f}), "
          f"{op.ledger.total_bits / 1e6:.3f} Mbit")
