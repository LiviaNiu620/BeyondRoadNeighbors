"""Feasible variable-metric projections for coordinated AV path flows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ProblemData
from .sensitivity import ReducedCoordinates

FloatArray = NDArray[np.float64]


def weighted_lower_simplex_projection(
    point: ArrayLike,
    total: float,
    lower_bound: float | ArrayLike = 0.0,
    metric: ArrayLike | None = None,
    tolerance: float = 1e-13,
) -> FloatArray:
    """Project onto ``sum(x)=total, x>=lower`` in a diagonal metric.

    The returned vector solves

    ``min 0.5 * sum_i metric_i * (x_i-point_i)^2``

    subject to the lower-bounded simplex constraints.  The scalar dual root is
    monotone and is solved by safeguarded bisection; the final cleanup preserves
    the total to floating-point accuracy without violating the lower bound.
    """

    values = np.asarray(point, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("point must be a nonempty finite vector")
    lower = np.broadcast_to(np.asarray(lower_bound, dtype=np.float64), values.shape).copy()
    weights = (
        np.ones_like(values)
        if metric is None
        else np.broadcast_to(np.asarray(metric, dtype=np.float64), values.shape).copy()
    )
    if not np.all(np.isfinite(lower)) or np.any(lower < 0.0):
        raise ValueError("lower_bound must be finite and nonnegative")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("metric must be finite and strictly positive")
    total = float(total)
    if not np.isfinite(total) or total < float(np.sum(lower)) - tolerance:
        raise ValueError("simplex total is incompatible with the lower bounds")
    if abs(total - float(np.sum(lower))) <= tolerance:
        return lower

    def candidate(dual: float) -> FloatArray:
        return np.maximum(lower, values - dual / weights)

    # Bracket the unique dual root.  These analytical endpoints place every
    # component respectively above and at/below its lower bound.
    lower_dual = float(np.min(weights * (values - (total - np.sum(lower) + lower))))
    upper_dual = float(np.max(weights * (values - lower)))
    scale = max(1.0, abs(lower_dual), abs(upper_dual))
    lower_dual -= scale
    upper_dual += scale
    for _ in range(200):
        middle = 0.5 * (lower_dual + upper_dual)
        projected = candidate(middle)
        if float(np.sum(projected)) > total:
            lower_dual = middle
        else:
            upper_dual = middle
        if upper_dual - lower_dual <= tolerance * max(1.0, abs(middle)):
            break
    projected = candidate(0.5 * (lower_dual + upper_dual))
    error = total - float(np.sum(projected))
    free = np.flatnonzero(projected > lower + 10.0 * tolerance)
    if free.size == 0:
        free = np.asarray([int(np.argmax(projected - lower))])
    # Allocate the tiny root-solve remainder according to inverse curvature.
    inverse = 1.0 / weights[free]
    projected[free] += error * inverse / float(np.sum(inverse))
    # When the dual root is large, the bisection stops with an absolute remainder that the
    # allocation above can push past a coordinate sitting near its bound. Finish with an
    # exact active-set pass: clamp such coordinates and re-solve on the remaining free set.
    # (Revision 2026-11: the original code raised ArithmeticError here; runs that never
    # reached this branch are unaffected.)
    for _ in range(projected.size):
        if np.min(projected - lower) >= -100.0 * tolerance:
            break
        clamped = projected < lower
        projected[clamped] = lower[clamped]
        free = np.flatnonzero(~clamped & (projected > lower))
        if free.size == 0:
            raise ArithmeticError("weighted simplex projection violated its lower bound")
        inverse = 1.0 / weights[free]
        projected[free] += (total - float(np.sum(projected))) * inverse / float(np.sum(inverse))
    if np.min(projected - lower) < -100.0 * tolerance:
        raise ArithmeticError("weighted simplex projection violated its lower bound")
    projected = np.maximum(projected, lower)
    projected[free[0]] += total - float(np.sum(projected))
    return projected


def full_gradient_from_reduced(
    coordinates: ReducedCoordinates, reduced_gradient: ArrayLike
) -> FloatArray:
    """Return a groupwise full gradient equivalent on conservation directions.

    Reduced coordinates use ``e_p-e_reference`` columns.  Setting every
    reference component to zero therefore reconstructs a representative whose
    inner product with every group-conserving displacement is exact.
    """

    reduced = np.asarray(reduced_gradient, dtype=np.float64)
    if reduced.shape != (coordinates.dimension,) or not np.all(np.isfinite(reduced)):
        raise ValueError("reduced_gradient has an invalid shape or value")
    full = np.zeros(coordinates.problem.number_of_paths, dtype=np.float64)
    full[np.asarray(coordinates.nonreference_indices, dtype=int)] = reduced
    return full


def demand_scaled_av_metric(problem: ProblemData, full_gradient: ArrayLike) -> FloatArray:
    """Build a groupwise scalar metric that normalizes flow and gradient units."""

    gradient = np.asarray(full_gradient, dtype=np.float64)
    if gradient.shape != (problem.number_of_paths,):
        raise ValueError("full_gradient has an invalid shape")
    metric = np.empty_like(gradient)
    for group in problem.groups:
        indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
        centered = gradient[indices] - float(np.mean(gradient[indices]))
        rms = float(np.sqrt(np.mean(centered * centered)))
        metric[indices] = max(rms / max(group.av_demand, 1e-12), 1e-12)
    return metric


@dataclass(frozen=True, slots=True)
class ProjectedAVStep:
    flow: FloatArray
    reduced: FloatArray
    full_gradient: FloatArray
    metric: FloatArray
    displacement: FloatArray
    directional_derivative: float
    mapping_norm: float
    normalized_mapping_norm: float
    minimum_flow: float


def projected_av_step(
    coordinates: ReducedCoordinates,
    current_flow: ArrayLike,
    reduced_gradient: ArrayLike,
    step: float,
    minimum_flow: float,
    metric: ArrayLike | None = None,
) -> ProjectedAVStep:
    """Take one feasible groupwise variable-metric projected AV step."""

    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("step must be finite and positive")
    current = np.asarray(current_flow, dtype=np.float64)
    problem = coordinates.problem
    if current.shape != (problem.number_of_paths,):
        raise ValueError("current_flow has an invalid shape")
    full_gradient = full_gradient_from_reduced(coordinates, reduced_gradient)
    weights = (
        demand_scaled_av_metric(problem, full_gradient)
        if metric is None
        else np.asarray(metric, dtype=np.float64)
    )
    if weights.shape != current.shape or np.any(weights <= 0.0):
        raise ValueError("metric has an invalid shape or value")
    projected = np.empty_like(current)
    normalized_parts: list[FloatArray] = []
    for group in problem.groups:
        indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
        point = current[indices] - step * full_gradient[indices] / weights[indices]
        projected[indices] = weighted_lower_simplex_projection(
            point,
            group.av_demand,
            minimum_flow,
            weights[indices],
        )
        normalized_parts.append(
            (current[indices] - projected[indices])
            / (step * max(group.av_demand, 1e-12))
        )
    displacement = projected - current
    mapping = -displacement / step
    normalized = np.concatenate(normalized_parts) if normalized_parts else np.zeros(0)
    return ProjectedAVStep(
        flow=projected,
        reduced=coordinates.reduced_from_flow(projected),
        full_gradient=full_gradient,
        metric=np.array(weights, copy=True),
        displacement=displacement,
        directional_derivative=float(np.dot(full_gradient, displacement)),
        mapping_norm=float(np.linalg.norm(mapping)),
        normalized_mapping_norm=float(
            np.sqrt(np.mean(normalized * normalized)) if normalized.size else 0.0
        ),
        minimum_flow=float(np.min(projected)),
    )
