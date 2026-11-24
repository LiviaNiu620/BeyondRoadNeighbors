"""Pure numerical operators for the Study 1 restricted path model."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from .model import ProblemData

FloatArray = NDArray[np.float64]


def _vector(values: ArrayLike, expected: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


@lru_cache(maxsize=32)
def sparse_incidence_matrix(problem: ProblemData) -> sparse.csc_matrix:
    """Cache the immutable link-path incidence used by every follower iteration."""

    rows: list[int] = []
    columns: list[int] = []
    for path_index, path in enumerate(problem.paths):
        rows.extend(path.link_indices)
        columns.extend([path_index] * len(path.link_indices))
    return sparse.csc_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(problem.number_of_links, problem.number_of_paths),
    )


def incidence_matrix(problem: ProblemData) -> FloatArray:
    """Return the deterministic link-by-path incidence matrix."""

    return sparse_incidence_matrix(problem).toarray()


def link_flows_from_paths(
    problem: ProblemData, hdv_path_flow: ArrayLike, av_path_flow: ArrayLike
) -> FloatArray:
    hdv = _vector(hdv_path_flow, problem.number_of_paths, "hdv_path_flow")
    av = _vector(av_path_flow, problem.number_of_paths, "av_path_flow")
    return np.asarray(sparse_incidence_matrix(problem) @ (hdv + av)).ravel()


def bpr_link_costs(problem: ProblemData, link_flow: ArrayLike) -> FloatArray:
    flow = _vector(link_flow, problem.number_of_links, "link_flow")
    if np.any(flow < 0.0):
        raise ValueError("link_flow must be nonnegative")
    fft = np.asarray([link.free_flow_time for link in problem.links])
    capacity = np.asarray([link.capacity for link in problem.links])
    alpha = np.asarray([link.alpha for link in problem.links])
    beta = np.asarray([link.beta for link in problem.links])
    return fft * (1.0 + alpha * np.power(flow / capacity, beta))


def bpr_link_derivatives(problem: ProblemData, link_flow: ArrayLike) -> FloatArray:
    flow = _vector(link_flow, problem.number_of_links, "link_flow")
    if np.any(flow < 0.0):
        raise ValueError("link_flow must be nonnegative")
    fft = np.asarray([link.free_flow_time for link in problem.links])
    capacity = np.asarray([link.capacity for link in problem.links])
    alpha = np.asarray([link.alpha for link in problem.links])
    beta = np.asarray([link.beta for link in problem.links])
    return fft * alpha * beta * np.power(flow, beta - 1.0) / np.power(
        capacity, beta
    )


def bpr_link_capacity_derivatives(
    problem: ProblemData, link_flow: ArrayLike
) -> FloatArray:
    """Return partial derivatives of BPR travel time with respect to capacity."""

    flow = _vector(link_flow, problem.number_of_links, "link_flow")
    if np.any(flow < 0.0):
        raise ValueError("link_flow must be nonnegative")
    fft = np.asarray([link.free_flow_time for link in problem.links])
    capacity = np.asarray([link.capacity for link in problem.links])
    alpha = np.asarray([link.alpha for link in problem.links])
    beta = np.asarray([link.beta for link in problem.links])
    return (
        -fft
        * alpha
        * beta
        * np.power(flow / capacity, beta)
        / capacity
    )


def path_costs(problem: ProblemData, link_cost: ArrayLike) -> FloatArray:
    cost = _vector(link_cost, problem.number_of_links, "link_cost")
    return np.asarray(sparse_incidence_matrix(problem).T @ cost).ravel()


def total_system_travel_time(link_flow: ArrayLike, link_cost: ArrayLike) -> float:
    flow = np.asarray(link_flow, dtype=np.float64)
    cost = np.asarray(link_cost, dtype=np.float64)
    if flow.shape != cost.shape or flow.ndim != 1:
        raise ValueError("link_flow and link_cost must be equal-length vectors")
    if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(cost)):
        raise ValueError("TSTT inputs must be finite")
    return float(np.dot(flow, cost))
