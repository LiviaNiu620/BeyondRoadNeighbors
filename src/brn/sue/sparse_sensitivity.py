"""Sparse reduced-KKT factorization with shared multi-RHS adjoints."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu

from .model import ProblemData
from .operators import (
    bpr_link_costs,
    bpr_link_derivatives,
    link_flows_from_paths,
    path_costs,
    sparse_incidence_matrix,
)
from .sensitivity import (
    ImplicitGradientResult,
    ReducedCoordinates,
    RegretGradientResult,
)

FloatArray = NDArray[np.float64]


def sparse_reduced_basis(coordinates: ReducedCoordinates) -> sparse.csc_matrix:
    problem = coordinates.problem
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    column = 0
    for group in problem.groups:
        indices = problem.path_indices(group.group_id)
        reference = indices[-1]
        for path_index in indices[:-1]:
            rows.extend((path_index, reference))
            columns.extend((column, column))
            values.extend((1.0, -1.0))
            column += 1
    return sparse.csc_matrix(
        (values, (rows, columns)),
        shape=(problem.number_of_paths, coordinates.dimension),
    )


def _recommendation_jacobian_sparse(
    coordinates: ReducedCoordinates, reduced_logits: ArrayLike
) -> tuple[FloatArray, sparse.csc_matrix]:
    logits = np.asarray(reduced_logits, dtype=np.float64)
    if logits.shape != (coordinates.dimension,):
        raise ValueError("reduced_logits has an invalid shape")
    q = coordinates.recommendation_from_logits(logits)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    cursor = 0
    for group in coordinates.problem.groups:
        indices = coordinates.problem.path_indices(group.group_id)
        for local_column, selected_path in enumerate(indices[:-1]):
            selected_probability = q[selected_path]
            for local_row, path_index in enumerate(indices):
                indicator = 1.0 if local_row == local_column else 0.0
                value = q[path_index] * (indicator - selected_probability)
                if value != 0.0:
                    rows.append(path_index)
                    columns.append(cursor + local_column)
                    values.append(value)
        cursor += len(indices) - 1
    jacobian = sparse.csc_matrix(
        (values, (rows, columns)),
        shape=(coordinates.problem.number_of_paths, coordinates.dimension),
    )
    return q, jacobian


@dataclass(slots=True)
class SparseKKTDiagnostics:
    dimension: int
    nonzeros: int
    factorization_count: int = 1
    solve_calls: int = 0
    rhs_columns: int = 0
    maximum_relative_residual: float = 0.0
    factorization_seconds: float = 0.0
    solve_seconds: float = 0.0


class SparseKKTSystem:
    """One SuperLU factorization shared by scalar and multi-column adjoints."""

    def __init__(self, matrix: sparse.csc_matrix):
        self.matrix = matrix
        self._factor: SuperLU | None = None
        self.diagnostics = SparseKKTDiagnostics(
            dimension=matrix.shape[0], nonzeros=matrix.nnz
        )
        if matrix.shape[0] > 0:
            start = perf_counter()
            self._factor = splu(matrix)
            self.diagnostics.factorization_seconds = perf_counter() - start

    def solve_transpose(self, right_hand_side: ArrayLike) -> FloatArray:
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.matrix.shape[0]:
            raise ValueError("adjoint RHS has an invalid leading dimension")
        columns = 1 if rhs.ndim == 1 else rhs.shape[1]
        self.diagnostics.solve_calls += 1
        self.diagnostics.rhs_columns += columns
        if self.matrix.shape[0] == 0:
            solution = np.zeros_like(rhs)
        else:
            if self._factor is None:
                raise RuntimeError("sparse KKT factorization is unavailable")
            start = perf_counter()
            solution = np.asarray(self._factor.solve(rhs, trans="T"))
            self.diagnostics.solve_seconds += perf_counter() - start
        residual = self.matrix.T @ solution - rhs
        if rhs.ndim == 1:
            denominator = max(float(np.linalg.norm(rhs)), 1.0)
            relative = float(np.linalg.norm(residual) / denominator)
        else:
            denominator = np.maximum(np.linalg.norm(rhs, axis=0), 1.0)
            relative = float(
                np.max(np.linalg.norm(residual, axis=0) / denominator, initial=0.0)
            )
        self.diagnostics.maximum_relative_residual = max(
            self.diagnostics.maximum_relative_residual, relative
        )
        return solution


@dataclass(slots=True)
class SparseSensitivityContext:
    coordinates: ReducedCoordinates
    hdv_path_flow: FloatArray
    av_path_flow: FloatArray
    reduced_logits: FloatArray
    incidence: sparse.csc_matrix
    basis: sparse.csc_matrix
    link_reduced_basis: sparse.csc_matrix
    link_flow: FloatArray
    link_cost: FloatArray
    link_derivative: FloatArray
    av_partial: sparse.csc_matrix
    logit_partial: sparse.csc_matrix
    kkt: SparseKKTSystem
    assembly_seconds: float = 0.0

    @classmethod
    def create(
        cls,
        coordinates: ReducedCoordinates,
        hdv_path_flow: ArrayLike,
        av_path_flow: ArrayLike,
        reduced_logits: ArrayLike,
        recommendation_floor: float = 1e-12,
    ) -> "SparseSensitivityContext":
        assembly_start = perf_counter()
        problem = coordinates.problem
        f = np.asarray(hdv_path_flow, dtype=np.float64)
        x = np.asarray(av_path_flow, dtype=np.float64)
        logits = np.asarray(reduced_logits, dtype=np.float64)
        if f.shape != (problem.number_of_paths,) or x.shape != f.shape:
            raise ValueError("path flows have invalid shapes")
        if logits.shape != (coordinates.dimension,):
            raise ValueError("reduced_logits has an invalid shape")
        if np.any(f <= 0.0):
            raise ValueError("sparse KKT requires strictly positive HDV flows")

        incidence = sparse_incidence_matrix(problem)
        basis = sparse_reduced_basis(coordinates)
        link_basis = (incidence @ basis).tocsc()
        link_flow = link_flows_from_paths(problem, f, x)
        link_cost = bpr_link_costs(problem, link_flow)
        derivative = bpr_link_derivatives(problem, link_flow)
        theta = np.asarray(
            [problem.group(path.group_id).cost_sensitivity for path in problem.paths]
        )
        entropy_diagonal = 1.0 / (theta * f)
        congestion_reduced = link_basis.T @ sparse.diags(derivative) @ link_basis
        entropy_reduced = basis.T @ sparse.diags(entropy_diagonal) @ basis
        kkt_matrix = (congestion_reduced + entropy_reduced).tocsc()

        q, q_jacobian = _recommendation_jacobian_sparse(coordinates, logits)
        delta = np.asarray(
            [
                problem.group(path.group_id).recommendation_sensitivity
                for path in problem.paths
            ]
        )
        direct_q = -delta / (theta * (q + recommendation_floor))
        logit_partial = (
            basis.T @ sparse.diags(direct_q) @ q_jacobian
        ).tocsc()
        kkt = SparseKKTSystem(kkt_matrix)
        assembly_seconds = (
            perf_counter() - assembly_start - kkt.diagnostics.factorization_seconds
        )
        return cls(
            coordinates=coordinates,
            hdv_path_flow=f,
            av_path_flow=x,
            reduced_logits=logits,
            incidence=incidence,
            basis=basis,
            link_reduced_basis=link_basis,
            link_flow=link_flow,
            link_cost=link_cost,
            link_derivative=derivative,
            av_partial=congestion_reduced.tocsc(),
            logit_partial=logit_partial,
            kkt=kkt,
            assembly_seconds=max(0.0, assembly_seconds),
        )

    def tstt_gradient(self) -> ImplicitGradientResult:
        marginal = self.link_cost + self.link_flow * self.link_derivative
        reduced_direct = np.asarray(self.link_reduced_basis.T @ marginal).ravel()
        adjoint = self.kkt.solve_transpose(reduced_direct)
        av_gradient = reduced_direct - np.asarray(
            self.av_partial.T @ adjoint
        ).ravel()
        logit_gradient = -np.asarray(
            self.logit_partial.T @ adjoint
        ).ravel()
        return ImplicitGradientResult(
            av_reduced_gradient=av_gradient,
            recommendation_logit_gradient=logit_gradient,
            adjoint=adjoint,
            adjoint_relative_residual=self.kkt.diagnostics.maximum_relative_residual,
            kkt_condition_number=float("nan"),
        )

    def regret_gradients(
        self, active_group_indices: ArrayLike | None = None
    ) -> RegretGradientResult:
        """Return per-group regret gradients with exact PHR active-RHS screening.

        Rows not requested through ``active_group_indices`` are returned as exact
        zeros.  This is safe when their PHR coefficient is zero because those
        rows are multiplied by zero in the augmented-Lagrangian gradient.  The
        KKT factorization is shared with the objective adjoint and only the
        selected regret right-hand sides are solved.
        """

        problem = self.coordinates.problem
        group_count = len(problem.groups)
        dimension = self.coordinates.dimension
        if active_group_indices is None:
            selected = np.arange(group_count, dtype=int)
        else:
            selected = np.asarray(active_group_indices, dtype=int)
            if selected.ndim != 1:
                raise ValueError("active_group_indices must be one-dimensional")
            if selected.size and (
                int(np.min(selected)) < 0 or int(np.max(selected)) >= group_count
            ):
                raise ValueError("active_group_indices contains an invalid group")
            selected = np.unique(selected)
        selected_count = int(selected.size)
        reduced_follower = np.zeros((dimension, selected_count), dtype=np.float64)
        reduced_direct_av = np.zeros_like(reduced_follower)
        current_costs = path_costs(problem, self.link_cost)
        for column, group_index in enumerate(selected):
            group = problem.groups[int(group_index)]
            indices = np.asarray(problem.path_indices(group.group_id), dtype=int)
            weights = np.zeros(problem.number_of_paths, dtype=np.float64)
            weights[indices] = self.hdv_path_flow[indices] / group.hdv_demand
            group_link_load = np.asarray(self.incidence @ weights).ravel()
            direct_av = np.asarray(
                self.link_reduced_basis.T
                @ (self.link_derivative * group_link_load)
            ).ravel()
            direct_weight = np.zeros(problem.number_of_paths, dtype=np.float64)
            direct_weight[indices] = current_costs[indices] / group.hdv_demand
            reduced_direct_av[:, column] = direct_av
            reduced_follower[:, column] = direct_av + np.asarray(
                self.basis.T @ direct_weight
            ).ravel()
        adjoints = self.kkt.solve_transpose(reduced_follower)
        selected_av_gradients = (
            reduced_direct_av - np.asarray(self.av_partial.T @ adjoints)
        ).T
        selected_logit_gradients = -np.asarray(self.logit_partial.T @ adjoints).T
        residuals = self.kkt.matrix.T @ adjoints - reduced_follower
        scales = np.maximum(np.linalg.norm(reduced_follower, axis=0), 1.0)
        relative = (
            np.linalg.norm(residuals, axis=0) / scales
            if selected_count
            else np.zeros(0, dtype=np.float64)
        )
        av_gradients = np.zeros((group_count, dimension), dtype=np.float64)
        logit_gradients = np.zeros_like(av_gradients)
        all_adjoints = np.zeros((group_count, dimension), dtype=np.float64)
        all_relative = np.zeros(group_count, dtype=np.float64)
        if selected_count:
            av_gradients[selected, :] = selected_av_gradients
            logit_gradients[selected, :] = selected_logit_gradients
            all_adjoints[selected, :] = adjoints.T
            all_relative[selected] = relative
        return RegretGradientResult(
            group_ids=tuple(group.group_id for group in problem.groups),
            av_reduced_gradients=av_gradients,
            recommendation_logit_gradients=logit_gradients,
            adjoints=all_adjoints,
            adjoint_relative_residuals=all_relative,
            kkt_condition_number=float("nan"),
        )
