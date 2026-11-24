"""Restricted-path stochastic HDV equilibrium for fixed AV flow and guidance."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ProblemData
from .operators import (
    bpr_link_costs,
    link_flows_from_paths,
    path_costs,
)
from .state import AssignmentState

FloatArray = NDArray[np.float64]


def _path_vector(problem: ProblemData, values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (problem.number_of_paths,):
        raise ValueError(
            f"{name} must have shape ({problem.number_of_paths},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _stable_softmax(values: FloatArray) -> FloatArray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    denominator = float(np.sum(weights))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("softmax normalization is not finite and positive")
    return weights / denominator


@lru_cache(maxsize=32)
def _behavior_arrays(problem: ProblemData) -> tuple[FloatArray, ...]:
    group_position = {group.group_id: index for index, group in enumerate(problem.groups)}
    group_index = np.asarray(
        [group_position[path.group_id] for path in problem.paths], dtype=np.int64
    )
    preference = np.asarray([path.preference_constant for path in problem.paths])
    av_preference = np.asarray([path.av_preference_constant for path in problem.paths])
    theta = np.asarray(
        [problem.group(path.group_id).cost_sensitivity for path in problem.paths]
    )
    av_theta = np.asarray(
        [problem.group(path.group_id).av_cost_sensitivity for path in problem.paths]
    )
    delta = np.asarray(
        [problem.group(path.group_id).recommendation_sensitivity for path in problem.paths]
    )
    hdv_demand = np.asarray([group.hdv_demand for group in problem.groups])
    av_demand = np.asarray([group.av_demand for group in problem.groups])
    return (
        group_index,
        preference,
        av_preference,
        theta,
        av_theta,
        delta,
        hdv_demand,
        av_demand,
    )


def _group_softmax_flow(
    utility: FloatArray,
    group_index: NDArray[np.int64],
    group_demand: FloatArray,
) -> FloatArray:
    maximum = np.full(len(group_demand), -np.inf, dtype=np.float64)
    np.maximum.at(maximum, group_index, utility)
    weights = np.exp(utility - maximum[group_index])
    denominator = np.zeros(len(group_demand), dtype=np.float64)
    np.add.at(denominator, group_index, weights)
    if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("group softmax denominator is invalid")
    return group_demand[group_index] * weights / denominator[group_index]


def _fixed_point_map_validated(
    problem: ProblemData,
    hdv: FloatArray,
    av: FloatArray,
    q: FloatArray,
    recommendation_floor: float,
) -> FloatArray:
    link_flow = link_flows_from_paths(problem, hdv, av)
    current_path_cost = path_costs(problem, bpr_link_costs(problem, link_flow))
    (
        group_index,
        preference,
        _,
        theta,
        _,
        delta,
        hdv_demand,
        _,
    ) = _behavior_arrays(problem)
    utility = (
        preference
        + delta * np.log(q + recommendation_floor)
        - theta * current_path_cost
    )
    return _group_softmax_flow(utility, group_index, hdv_demand)


def uniform_recommendation(problem: ProblemData) -> FloatArray:
    """Return the no-guidance distribution, uniform within every group."""

    recommendation = np.zeros(problem.number_of_paths, dtype=np.float64)
    for group in problem.groups:
        indices = problem.path_indices(group.group_id)
        recommendation[np.asarray(indices, dtype=int)] = 1.0 / len(indices)
    return recommendation


def validate_group_simplex(
    problem: ProblemData, values: ArrayLike, name: str, tolerance: float = 1e-10
) -> FloatArray:
    vector = _path_vector(problem, values, name)
    if np.min(vector) < -tolerance:
        raise ValueError(f"{name} contains a negative component")
    for group in problem.groups:
        total = float(np.sum(vector[np.asarray(problem.path_indices(group.group_id))]))
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"{name} for group {group.group_id} sums to {total}, not one"
            )
    return vector


def validate_av_flow(
    problem: ProblemData, values: ArrayLike, tolerance: float = 1e-10
) -> FloatArray:
    flow = _path_vector(problem, values, "av_path_flow")
    if np.min(flow) < -tolerance:
        raise ValueError("av_path_flow contains a negative component")
    for group in problem.groups:
        indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
        total = float(np.sum(flow[indices]))
        if abs(total - group.av_demand) > tolerance:
            raise ValueError(
                f"AV flow for group {group.group_id} is {total}, "
                f"expected {group.av_demand}"
            )
    return flow


def initial_hdv_flow(problem: ProblemData) -> FloatArray:
    """Return a deterministic demand-feasible uniform HDV flow."""

    flow = np.zeros(problem.number_of_paths, dtype=np.float64)
    for group in problem.groups:
        indices = problem.path_indices(group.group_id)
        flow[np.asarray(indices, dtype=int)] = group.hdv_demand / len(indices)
    return flow


def validate_hdv_flow(
    problem: ProblemData, values: ArrayLike, tolerance: float = 1e-8
) -> FloatArray:
    flow = _path_vector(problem, values, "hdv_path_flow")
    if np.min(flow) < -tolerance:
        raise ValueError("hdv_path_flow contains a negative component")
    for group in problem.groups:
        indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
        total = float(np.sum(flow[indices]))
        if abs(total - group.hdv_demand) > tolerance:
            raise ValueError(
                f"HDV flow for group {group.group_id} is {total}, "
                f"expected {group.hdv_demand}"
            )
    return flow


def fixed_point_map(
    problem: ProblemData,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    recommendation: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> FloatArray:
    """Evaluate the controlled Logit best-response map at the supplied flow."""

    if recommendation_floor <= 0.0:
        raise ValueError("recommendation_floor must be positive")
    hdv = validate_hdv_flow(problem, hdv_path_flow)
    av = validate_av_flow(problem, av_path_flow)
    q = validate_group_simplex(problem, recommendation, "recommendation")

    return _fixed_point_map_validated(
        problem, hdv, av, q, recommendation_floor
    )


@dataclass(frozen=True, slots=True)
class SolverOptions:
    tolerance: float = 1e-8
    max_iterations: int = 10_000
    minimum_iterations: int = 1
    msa_exponent: float = 0.7
    recommendation_floor: float = 1e-12
    warm_start_iteration_offset: int = 100

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.minimum_iterations < 1:
            raise ValueError("minimum_iterations must be positive")
        if self.minimum_iterations > self.max_iterations:
            raise ValueError("minimum_iterations exceeds max_iterations")
        if self.warm_start_iteration_offset < 0:
            raise ValueError("warm_start_iteration_offset must be nonnegative")
        if not 0.5 < self.msa_exponent <= 1.0:
            raise ValueError("msa_exponent must be in (0.5, 1]")
        if self.recommendation_floor <= 0.0:
            raise ValueError("recommendation_floor must be positive")


@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    state: AssignmentState
    converged: bool
    iterations: int
    residual: float
    residual_history: tuple[float, ...]
    termination_reason: str


class LogitEquilibriumSolver:
    """Deterministic MSA solver with an explicit fixed-point residual."""

    def __init__(self, options: SolverOptions | None = None):
        self.options = options or SolverOptions()

    def solve(
        self,
        problem: ProblemData,
        av_path_flow: ArrayLike,
        recommendation: ArrayLike | None = None,
        start: ArrayLike | None = None,
    ) -> EquilibriumResult:
        av = np.array(validate_av_flow(problem, av_path_flow), copy=True)
        q = np.array(
            validate_group_simplex(
                problem,
                uniform_recommendation(problem)
                if recommendation is None
                else recommendation,
                "recommendation",
            ),
            copy=True,
        )
        hdv = np.array(
            initial_hdv_flow(problem)
            if start is None
            else validate_hdv_flow(problem, start),
            copy=True,
        )

        history: list[float] = []
        converged = False
        iterations = 0
        demand_scale = max(problem.total_hdv_demand, 1.0)
        iteration_offset = (
            self.options.warm_start_iteration_offset if start is not None else 0
        )

        for iteration in range(1, self.options.max_iterations + 1):
            response = _fixed_point_map_validated(
                problem,
                hdv,
                av,
                q,
                self.options.recommendation_floor,
            )
            residual = float(np.linalg.norm(hdv - response, ord=1) / demand_scale)
            history.append(residual)
            iterations = iteration
            if (
                iteration >= self.options.minimum_iterations
                and residual <= self.options.tolerance
            ):
                converged = True
                break

            step = 1.0 / (
                (iteration + iteration_offset) ** self.options.msa_exponent
            )
            hdv = (1.0 - step) * hdv + step * response

        link_flow = link_flows_from_paths(problem, hdv, av)
        link_cost = bpr_link_costs(problem, link_flow)
        current_path_cost = path_costs(problem, link_cost)
        final_response = _fixed_point_map_validated(
            problem,
            hdv,
            av,
            q,
            self.options.recommendation_floor,
        )
        final_residual = float(
            np.linalg.norm(hdv - final_response, ord=1) / demand_scale
        )
        if not history or final_residual != history[-1]:
            history.append(final_residual)
        if final_residual <= self.options.tolerance:
            converged = True

        state = AssignmentState(
            problem=problem,
            hdv_path_flow=hdv,
            av_path_flow=av,
            recommendation=q,
            link_flow=link_flow,
            link_cost=link_cost,
            path_cost=current_path_cost,
        )
        return EquilibriumResult(
            state=state,
            converged=converged,
            iterations=iterations,
            residual=final_residual,
            residual_history=tuple(history),
            termination_reason="converged" if converged else "maximum_iterations",
        )
