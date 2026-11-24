"""Logit stochastic-user-equilibrium solver and implicit-differentiation utilities.

Vendored subset of ``code/study1`` from the authors' Mixed_Autonomy_Traffic_Assignment
project (only the modules this repository uses). The weighted-simplex projection carries the
exact active-set finish described in Appendix C of the paper.
"""

from .equilibrium import (
    EquilibriumResult,
    LogitEquilibriumSolver,
    SolverOptions,
    fixed_point_map,
    uniform_recommendation,
)
from .model import DemandGroup, Link, Path, ProblemData
from .projection import ProjectedAVStep, projected_av_step, weighted_lower_simplex_projection
from .sensitivity import ReducedCoordinates

__all__ = [
    "DemandGroup",
    "EquilibriumResult",
    "Link",
    "LogitEquilibriumSolver",
    "Path",
    "ProblemData",
    "ProjectedAVStep",
    "ReducedCoordinates",
    "SolverOptions",
    "fixed_point_map",
    "projected_av_step",
    "uniform_recommendation",
    "weighted_lower_simplex_projection",
]
