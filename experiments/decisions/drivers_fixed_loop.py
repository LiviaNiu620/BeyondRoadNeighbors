"""Equilibrium-unaware coordination with full coverage: authorities treat drivers as fixed.

Gradient oracle: managed-flow gradient = b (direct marginal-cost derivative, all
regions' contributions), guidance gradient = 0 (guidance acts only through drivers).
Same outer rule, start and attempts as run_v2. Also reports, at the start state, the
share of the adjoint due to the drivers' response, ||lambda - N^-1 b||_N / ||N^-1 b||_N,
against the bound gamma/(1+gamma).
"""
import argparse, json, time, sys
import numpy as np
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator
from brn.experiments.common import *  # noqa
from brn.experiments import demand as scaled
from brn.spectral import entropy_preconditioner
from brn.sue.operators import bpr_link_costs, link_flows_from_paths, total_system_travel_time
from brn.sue.projection import projected_av_step
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=16); ap.add_argument("--theta", type=float, default=0.08)
ap.add_argument("--demand-scale", type=float, default=1.0); ap.add_argument("--outer", type=int, default=30)
ap.add_argument("--out", required=True); args = ap.parse_args()
scaled.apply(args.demand_scale)
problem, coords, av0, lg0 = load(theta=args.theta)
dec = regions(problem, coords, args.k)
solver = LogitEquilibriumSolver(SolverOptions(tolerance=1e-10, max_iterations=200_000, msa_exponent=0.7))
def evaluate(av, lg):
    eq = solver.solve(problem, av_path_flow=av, recommendation=coords.recommendation_from_logits(lg))
    h = eq.state.hdv_path_flow; lf = link_flows_from_paths(problem, h, av)
    return h, float(total_system_travel_time(lf, bpr_link_costs(problem, lf))), eq.converged
hdv, tstt, ok = evaluate(av0, lg0); init = tstt
S = assemble_local_systems(problem, coords, dec, hdv, av0); A, b = centralized_system(S)
P = entropy_preconditioner(problem, hdv); N = P.matrix()
lam = spsolve(A.tocsc(), b); lam0 = P.apply(b)
nN = lambda x: float(np.sqrt(x @ (N @ x)))
Minv = LinearOperator(N.shape, matvec=P.apply, dtype=float)
gam = float(eigsh(A, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-6)[0]) - 1
share = nN(lam - lam0) / nN(lam0)
print(f"gamma={gam:.3f} response share={share:.3f} bound={gam/(1+gam):.3f}", flush=True)
av, lg = np.array(av0, float), np.array(lg0, float); av_step = 1.0; best = tstt; acc = 0; t0 = time.perf_counter()
for it in range(args.outer):
    S = assemble_local_systems(problem, coords, dec, hdv, av); _, b = centralized_system(S)
    step = projected_av_step(coords, av, b, av_step, 1e-6)
    nh, nt, ok = evaluate(step.flow, lg)
    if (not ok) or nt > tstt:
        av_step *= 0.5
        if av_step < 1e-10: break
        continue
    av, hdv, tstt = step.flow, nh, nt; acc += 1; best = min(best, tstt)
rec = dict(config="unaware-full", theta=args.theta, demand_scale=args.demand_scale, n_regions=args.k,
           gamma_exact=gam, response_share=share, response_bound=gam / (1 + gam),
           outer_iterations=acc, tstt=best, initial_tstt=init, tstt_improvement=best / init - 1, seconds=time.perf_counter() - t0)
open(args.out, "w").write(json.dumps(rec) + "\n")
print(rec, flush=True)
