"""Shared instance definition for every experiment: network, drivers, AV share, partition."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from brn import (assemble_local_systems, build_partition, centralized_system,  # noqa
                    decompose, reference_controls, tntp_dir)
from brn.sue.equilibrium import LogitEquilibriumSolver, SolverOptions  # noqa
from brn.sue.io.tntp import assemble_tntp_problem  # noqa
from brn.sue.sensitivity import ReducedCoordinates  # noqa
from brn.paths import OUTPUTS  # noqa  (default output directory for new runs)


def load(network="Anaheim", theta=0.08, av_share=0.25, delta=0.5, paths=3):
    d = tntp_dir()
    problem = assemble_tntp_problem(d / f"{network}_net.tntp", d / f"{network}_trips.tntp",
                                    av_share=av_share, paths_per_group=paths,
                                    cost_sensitivity=theta, av_cost_sensitivity=theta,
                                    recommendation_sensitivity=delta)
    coords = ReducedCoordinates.create(problem)
    a, l = reference_controls(problem, coords)
    return problem, coords, coords.flow_from_reduced(a, "av"), l


def follower(problem, coords, av, logits, tol=1e-10):
    eq = LogitEquilibriumSolver(SolverOptions(tolerance=tol, max_iterations=200_000,
                                              msa_exponent=0.7)).solve(
        problem, av_path_flow=av, recommendation=coords.recommendation_from_logits(logits))
    if not eq.converged:
        raise RuntimeError("follower did not converge")
    return eq.state.hdv_path_flow


def regions(problem, coords, K):
    return decompose(problem, coords, build_partition(problem, K, "spectral"), K)
