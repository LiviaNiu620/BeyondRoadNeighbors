"""Entropy potential, reduced KKT system, and dense adjoint differentiation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ProblemData
from .operators import (
    bpr_link_capacity_derivatives,
    bpr_link_costs,
    bpr_link_derivatives,
    incidence_matrix,
    link_flows_from_paths,
    path_costs,
    total_system_travel_time,
)

FloatArray = NDArray[np.float64]


def _condition_number(matrix: FloatArray) -> float:
    """Treat the vacuous zero-dimensional reduced system as perfectly conditioned."""

    return 1.0 if matrix.shape == (0, 0) else float(np.linalg.cond(matrix))


def _vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


@dataclass(frozen=True, slots=True)
class ReducedCoordinates:
    """Deterministic group-conservation basis with one reference path per group."""

    problem: ProblemData
    basis: FloatArray
    nonreference_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]

    @classmethod
    def create(cls, problem: ProblemData) -> "ReducedCoordinates":
        nonreference: list[int] = []
        references: list[int] = []
        columns: list[FloatArray] = []
        for group in problem.groups:
            indices = problem.path_indices(group.group_id)
            reference = indices[-1]
            references.append(reference)
            for path_index in indices[:-1]:
                column = np.zeros(problem.number_of_paths, dtype=np.float64)
                column[path_index] = 1.0
                column[reference] = -1.0
                columns.append(column)
                nonreference.append(path_index)
        basis = (
            np.column_stack(columns)
            if columns
            else np.zeros((problem.number_of_paths, 0), dtype=np.float64)
        )
        basis.setflags(write=False)
        return cls(
            problem=problem,
            basis=basis,
            nonreference_indices=tuple(nonreference),
            reference_indices=tuple(references),
        )

    @property
    def dimension(self) -> int:
        return self.basis.shape[1]

    def flow_from_reduced(self, reduced: ArrayLike, population: str) -> FloatArray:
        values = _vector(reduced, self.dimension, "reduced_flow")
        flow = np.zeros(self.problem.number_of_paths, dtype=np.float64)
        cursor = 0
        for group in self.problem.groups:
            indices = self.problem.path_indices(group.group_id)
            demand = (
                group.hdv_demand if population == "hdv" else group.av_demand
            )
            group_nonreference = len(indices) - 1
            if group_nonreference:
                flow[np.asarray(indices[:-1], dtype=int)] = values[
                    cursor : cursor + group_nonreference
                ]
            flow[indices[-1]] = demand - float(
                np.sum(values[cursor : cursor + group_nonreference])
            )
            cursor += group_nonreference
        if population not in {"hdv", "av"}:
            raise ValueError("population must be 'hdv' or 'av'")
        return flow

    def reduced_from_flow(self, flow: ArrayLike) -> FloatArray:
        values = _vector(flow, self.problem.number_of_paths, "path_flow")
        return np.array(values[np.asarray(self.nonreference_indices, dtype=int)], copy=True)

    def recommendation_from_logits(self, reduced_logits: ArrayLike) -> FloatArray:
        logits = _vector(reduced_logits, self.dimension, "reduced_logits")
        recommendation = np.zeros(self.problem.number_of_paths, dtype=np.float64)
        cursor = 0
        for group in self.problem.groups:
            indices = self.problem.path_indices(group.group_id)
            count = len(indices) - 1
            group_logits = np.zeros(len(indices), dtype=np.float64)
            if count:
                group_logits[:-1] = logits[cursor : cursor + count]
            shifted = group_logits - np.max(group_logits)
            weights = np.exp(shifted)
            recommendation[np.asarray(indices, dtype=int)] = weights / np.sum(weights)
            cursor += count
        return recommendation

    def recommendation_jacobian(self, reduced_logits: ArrayLike) -> FloatArray:
        q = self.recommendation_from_logits(reduced_logits)
        jacobian = np.zeros(
            (self.problem.number_of_paths, self.dimension), dtype=np.float64
        )
        cursor = 0
        for group in self.problem.groups:
            indices = np.asarray(self.problem.path_indices(group.group_id), dtype=int)
            count = len(indices) - 1
            for local_column in range(count):
                selected_probability = q[indices[local_column]]
                for local_row, path_index in enumerate(indices):
                    indicator = 1.0 if local_row == local_column else 0.0
                    jacobian[path_index, cursor + local_column] = q[path_index] * (
                        indicator - selected_probability
                    )
            cursor += count
        return jacobian

    def logits_from_recommendation(self, recommendation: ArrayLike) -> FloatArray:
        """Invert the reference-normalized softmax on a strictly positive simplex."""

        q = _vector(
            recommendation, self.problem.number_of_paths, "recommendation"
        )
        if np.any(q <= 0.0):
            raise ValueError("recommendation inversion requires positive probabilities")
        logits = np.zeros(self.dimension, dtype=np.float64)
        cursor = 0
        for group in self.problem.groups:
            indices = np.asarray(self.problem.path_indices(group.group_id), dtype=int)
            total = float(np.sum(q[indices]))
            if abs(total - 1.0) > 1e-10:
                raise ValueError(
                    f"recommendation for group {group.group_id} does not sum to one"
                )
            reference = q[indices[-1]]
            count = len(indices) - 1
            if count:
                logits[cursor : cursor + count] = np.log(q[indices[:-1]] / reference)
            cursor += count
        return logits


def _path_parameters(problem: ProblemData) -> tuple[FloatArray, FloatArray, FloatArray]:
    theta = np.zeros(problem.number_of_paths, dtype=np.float64)
    delta = np.zeros(problem.number_of_paths, dtype=np.float64)
    demand = np.zeros(problem.number_of_paths, dtype=np.float64)
    for index, path in enumerate(problem.paths):
        group = problem.group(path.group_id)
        theta[index] = group.cost_sensitivity
        delta[index] = group.recommendation_sensitivity
        demand[index] = group.hdv_demand
    return theta, delta, demand


def entropy_potential(
    problem: ProblemData,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    recommendation: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> float:
    f = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    x = _vector(av_path_flow, problem.number_of_paths, "av_path_flow")
    q = _vector(recommendation, problem.number_of_paths, "recommendation")
    if np.any(f < 0.0) or np.any(x < 0.0) or np.any(q < 0.0):
        raise ValueError("potential inputs must be nonnegative")
    link_flow = link_flows_from_paths(problem, f, x)
    integral = 0.0
    for value, link in zip(link_flow, problem.links):
        integral += link.free_flow_time * (
            value
            + link.alpha
            * value ** (link.beta + 1.0)
            / ((link.beta + 1.0) * link.capacity**link.beta)
        )
    theta, delta, demand = _path_parameters(problem)
    preference = np.asarray([path.preference_constant for path in problem.paths])
    positive = f > 0.0
    entropy = np.zeros_like(f)
    entropy[positive] = f[positive] * (
        np.log(f[positive] / demand[positive])
        - 1.0
        - preference[positive]
        - delta[positive] * np.log(q[positive] + recommendation_floor)
    ) / theta[positive]
    return float(integral + np.sum(entropy))


def follower_gradient(
    problem: ProblemData,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    recommendation: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> FloatArray:
    f = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    x = _vector(av_path_flow, problem.number_of_paths, "av_path_flow")
    q = _vector(recommendation, problem.number_of_paths, "recommendation")
    if np.any(f <= 0.0):
        raise ValueError("reduced KKT requires strictly positive HDV path flow")
    link_flow = link_flows_from_paths(problem, f, x)
    costs = path_costs(problem, bpr_link_costs(problem, link_flow))
    theta, delta, demand = _path_parameters(problem)
    preference = np.asarray([path.preference_constant for path in problem.paths])
    return costs + (
        np.log(f / demand)
        - preference
        - delta * np.log(q + recommendation_floor)
    ) / theta


def reduced_kkt_residual(
    coordinates: ReducedCoordinates,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    recommendation: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> FloatArray:
    gradient = follower_gradient(
        coordinates.problem,
        hdv_path_flow,
        av_path_flow,
        recommendation,
        recommendation_floor,
    )
    return coordinates.basis.T @ gradient


def congestion_path_hessian(
    problem: ProblemData, hdv_path_flow: ArrayLike, av_path_flow: ArrayLike
) -> FloatArray:
    link_flow = link_flows_from_paths(problem, hdv_path_flow, av_path_flow)
    derivatives = bpr_link_derivatives(problem, link_flow)
    incidence = incidence_matrix(problem)
    return incidence.T @ (derivatives[:, None] * incidence)


def reduced_kkt_jacobian(
    coordinates: ReducedCoordinates,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
) -> FloatArray:
    problem = coordinates.problem
    f = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    if np.any(f <= 0.0):
        raise ValueError("reduced KKT Jacobian requires positive HDV flow")
    theta, _, _ = _path_parameters(problem)
    full_hessian = congestion_path_hessian(problem, f, av_path_flow) + np.diag(
        1.0 / (theta * f)
    )
    return coordinates.basis.T @ full_hessian @ coordinates.basis


def reduced_control_partials(
    coordinates: ReducedCoordinates,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    reduced_logits: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> tuple[FloatArray, FloatArray]:
    problem = coordinates.problem
    q = coordinates.recommendation_from_logits(reduced_logits)
    congestion = congestion_path_hessian(problem, hdv_path_flow, av_path_flow)
    av_partial = coordinates.basis.T @ congestion @ coordinates.basis
    theta, delta, _ = _path_parameters(problem)
    q_jacobian = coordinates.recommendation_jacobian(reduced_logits)
    direct_q = -delta / (theta * (q + recommendation_floor))
    logit_partial = coordinates.basis.T @ (
        direct_q[:, None] * q_jacobian
    )
    return av_partial, logit_partial


@dataclass(frozen=True, slots=True)
class ImplicitGradientResult:
    av_reduced_gradient: FloatArray
    recommendation_logit_gradient: FloatArray
    adjoint: FloatArray
    adjoint_relative_residual: float
    kkt_condition_number: float


@dataclass(frozen=True, slots=True)
class CapacityGradientResult:
    """Implicit TSTT derivative with respect to positive surviving capacities."""

    capacity_gradient: FloatArray
    direct_capacity_gradient: FloatArray
    adjoint: FloatArray
    adjoint_relative_residual: float
    kkt_condition_number: float


@dataclass(frozen=True, slots=True)
class RegretGradientResult:
    """Per-group derivatives of average HDV cost (and hence average regret).

    Baseline costs and regret limits are constants, so these are also the exact
    derivatives of ``average_cost - baseline_average_cost - limit``.
    Rows follow ``problem.groups`` and columns follow the reduced AV/logit
    coordinates.
    """

    group_ids: tuple[str, ...]
    av_reduced_gradients: FloatArray
    recommendation_logit_gradients: FloatArray
    adjoints: FloatArray
    adjoint_relative_residuals: FloatArray
    kkt_condition_number: float


def implicit_tstt_gradient(
    coordinates: ReducedCoordinates,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    reduced_logits: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> ImplicitGradientResult:
    problem = coordinates.problem
    f = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    x = _vector(av_path_flow, problem.number_of_paths, "av_path_flow")
    jacobian = reduced_kkt_jacobian(coordinates, f, x)
    link_flow = link_flows_from_paths(problem, f, x)
    link_cost = bpr_link_costs(problem, link_flow)
    marginal = link_cost + link_flow * bpr_link_derivatives(problem, link_flow)
    full_path_derivative = incidence_matrix(problem).T @ marginal
    reduced_follower_derivative = coordinates.basis.T @ full_path_derivative
    adjoint = np.linalg.solve(jacobian.T, reduced_follower_derivative)
    av_partial, logit_partial = reduced_control_partials(
        coordinates,
        f,
        x,
        reduced_logits,
        recommendation_floor,
    )
    direct_av = coordinates.basis.T @ full_path_derivative
    av_gradient = direct_av - av_partial.T @ adjoint
    logit_gradient = -logit_partial.T @ adjoint
    linear_residual = jacobian.T @ adjoint - reduced_follower_derivative
    residual_scale = max(float(np.linalg.norm(reduced_follower_derivative)), 1.0)
    return ImplicitGradientResult(
        av_reduced_gradient=av_gradient,
        recommendation_logit_gradient=logit_gradient,
        adjoint=adjoint,
        adjoint_relative_residual=float(np.linalg.norm(linear_residual) / residual_scale),
        kkt_condition_number=_condition_number(jacobian),
    )


def implicit_tstt_capacity_gradient(
    coordinates: ReducedCoordinates,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
) -> CapacityGradientResult:
    """Differentiate reduced TSTT with respect to each positive link capacity."""

    problem = coordinates.problem
    f = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    x = _vector(av_path_flow, problem.number_of_paths, "av_path_flow")
    jacobian = reduced_kkt_jacobian(coordinates, f, x)
    incidence = incidence_matrix(problem)
    link_flow = link_flows_from_paths(problem, f, x)
    link_cost = bpr_link_costs(problem, link_flow)
    marginal = link_cost + link_flow * bpr_link_derivatives(problem, link_flow)
    full_path_derivative = incidence.T @ marginal
    follower_rhs = coordinates.basis.T @ full_path_derivative
    adjoint = np.linalg.solve(jacobian.T, follower_rhs)
    link_capacity_partial = bpr_link_capacity_derivatives(problem, link_flow)
    follower_capacity_partial = (
        coordinates.basis.T
        @ incidence.T
        @ np.diag(link_capacity_partial)
    )
    direct = link_flow * link_capacity_partial
    gradient = direct - follower_capacity_partial.T @ adjoint
    linear_residual = jacobian.T @ adjoint - follower_rhs
    residual_scale = max(float(np.linalg.norm(follower_rhs)), 1.0)
    return CapacityGradientResult(
        capacity_gradient=gradient,
        direct_capacity_gradient=direct,
        adjoint=adjoint,
        adjoint_relative_residual=float(
            np.linalg.norm(linear_residual) / residual_scale
        ),
        kkt_condition_number=_condition_number(jacobian),
    )


def implicit_average_regret_gradients(
    coordinates: ReducedCoordinates,
    hdv_path_flow: ArrayLike,
    av_path_flow: ArrayLike,
    reduced_logits: ArrayLike,
    recommendation_floor: float = 1e-12,
) -> RegretGradientResult:
    """Differentiate every group average-regret constraint by dense adjoints.

    For group ``g``, with ``w_g[p] = f[p] / d_g`` on its paths,

    ``d cbar_g / df = S_g C / d_g + H_c w_g`` and
    ``d cbar_g / dx = H_c w_g``.

    Conservation is already eliminated by ``coordinates.basis``.  One adjoint
    right-hand side is solved per group; the KKT factor is shared conceptually
    (and NumPy solves all right-hand sides in one dense call).
    """

    problem = coordinates.problem
    f = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    x = _vector(av_path_flow, problem.number_of_paths, "av_path_flow")
    logits = _vector(reduced_logits, coordinates.dimension, "reduced_logits")
    if any(group.hdv_demand <= 0.0 for group in problem.groups):
        raise ValueError("average-regret gradients require positive HDV demand")

    jacobian = reduced_kkt_jacobian(coordinates, f, x)
    congestion = congestion_path_hessian(problem, f, x)
    link_flow = link_flows_from_paths(problem, f, x)
    costs = path_costs(problem, bpr_link_costs(problem, link_flow))
    av_partial, logit_partial = reduced_control_partials(
        coordinates, f, x, logits, recommendation_floor
    )

    group_count = len(problem.groups)
    full_follower_derivatives = np.zeros(
        (problem.number_of_paths, group_count), dtype=np.float64
    )
    full_direct_av = np.zeros_like(full_follower_derivatives)
    for column, group in enumerate(problem.groups):
        indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
        weights = np.zeros(problem.number_of_paths, dtype=np.float64)
        weights[indices] = f[indices] / group.hdv_demand
        congestion_effect = congestion @ weights
        direct_weight = np.zeros(problem.number_of_paths, dtype=np.float64)
        direct_weight[indices] = costs[indices] / group.hdv_demand
        full_follower_derivatives[:, column] = direct_weight + congestion_effect
        full_direct_av[:, column] = congestion_effect

    reduced_follower_derivatives = coordinates.basis.T @ full_follower_derivatives
    adjoints = np.linalg.solve(jacobian.T, reduced_follower_derivatives)
    direct_av = coordinates.basis.T @ full_direct_av
    av_gradients = (direct_av - av_partial.T @ adjoints).T
    logit_gradients = (-logit_partial.T @ adjoints).T
    linear_residuals = jacobian.T @ adjoints - reduced_follower_derivatives
    residual_norms = np.linalg.norm(linear_residuals, axis=0)
    residual_scales = np.maximum(
        np.linalg.norm(reduced_follower_derivatives, axis=0), 1.0
    )
    return RegretGradientResult(
        group_ids=tuple(group.group_id for group in problem.groups),
        av_reduced_gradients=av_gradients,
        recommendation_logit_gradients=logit_gradients,
        adjoints=adjoints.T,
        adjoint_relative_residuals=residual_norms / residual_scales,
        kkt_condition_number=_condition_number(jacobian),
    )


def tstt_at_flows(
    problem: ProblemData, hdv_path_flow: ArrayLike, av_path_flow: ArrayLike
) -> float:
    link_flow = link_flows_from_paths(problem, hdv_path_flow, av_path_flow)
    return total_system_travel_time(link_flow, bpr_link_costs(problem, link_flow))
