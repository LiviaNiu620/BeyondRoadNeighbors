"""Reduction-free distributed adjoint with an a-priori spectral certificate (Theorem 4, Proposition 6).

The reduced follower Jacobian splits as  A = C + N  with
    C = E^T D^T T' D E   (congestion, PSD, sum of regional link terms)
    N = E^T H E          (logit entropy, SPD, block diagonal by OD group).

N is owned: every block belongs to the region that owns the group, so each
authority can apply N^{-1} to its own coordinates with no communication
(Sherman-Morrison on  diag(h_nonref) + h_ref 11^T).  With P = N,

    spec(P^{-1} A) subset [1, 1 + gamma],
    gamma = max_p sum_{a in p} t'_a(v_a) w_a,   w_a = sum_{r ni a} theta_r f_r,

(Cauchy-Schwarz on each link).  gamma is a theta-weighted path externality.
Its regional relaxation gamma_reg = sum_k max_p sum_{a in p cap A_k} t'_a w_a
needs only one scalar from each authority.

Given a spectral interval, the second-order stationary Richardson iteration
(stationary Chebyshev / heavy ball with optimal parameters) needs no inner
products: no global reductions and no residual-based stopping rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .distributed import AdjointSolution, DistributedOperator

FloatArray = NDArray[np.float64]


@dataclass
class EntropyPreconditioner:
    """Applies N^{-1} block by block. Each block is local to its owner."""

    starts: NDArray[np.int_]      # first reduced column of each group
    sizes: NDArray[np.int_]       # number of reduced columns of each group
    h_nonref: FloatArray          # 1/(theta f) of the nonreference path, per column
    h_ref: FloatArray             # 1/(theta f) of the reference path, per group

    def apply(self, v: FloatArray) -> FloatArray:
        v = np.asarray(v, dtype=np.float64)
        block = v.ndim == 2
        x = v if block else v[:, None]
        dinv = 1.0 / self.h_nonref
        y = dinv[:, None] * x
        # Sherman-Morrison per group: (D + h 11^T)^{-1} x = y - h (1^T y)/(1 + h 1^T D^{-1} 1) D^{-1}1
        group_of = np.repeat(np.arange(self.sizes.size), self.sizes)
        sum_y = np.zeros((self.sizes.size, x.shape[1]))
        np.add.at(sum_y, group_of, y)
        sum_d = np.zeros(self.sizes.size)
        np.add.at(sum_d, group_of, dinv)
        coef = self.h_ref / (1.0 + self.h_ref * sum_d)
        y = y - dinv[:, None] * (coef[:, None] * sum_y)[group_of]
        return y if block else y[:, 0]

    def matrix(self):
        from scipy import sparse
        group_of = np.repeat(np.arange(self.sizes.size), self.sizes)
        rows, cols, vals = [], [], []
        for g, (s, m) in enumerate(zip(self.starts, self.sizes)):
            idx = np.arange(s, s + m)
            r, c = np.meshgrid(idx, idx, indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel())
            vals.append(np.full(m * m, self.h_ref[g]))
        n = self.h_nonref.size
        M = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                              np.concatenate(cols))), shape=(n, n)).tocsr()
        return (M + sparse.diags(self.h_nonref)).tocsr()


def entropy_preconditioner(problem, hdv_path_flow) -> EntropyPreconditioner:
    group_by_id = {g.group_id: g for g in problem.groups}
    theta = np.asarray([group_by_id[p.group_id].cost_sensitivity for p in problem.paths])
    h = 1.0 / (theta * np.asarray(hdv_path_flow, dtype=np.float64))
    starts, sizes, h_nonref, h_ref = [], [], [], []
    column = 0
    for group in problem.groups:
        indices = problem.path_indices(group.group_id)
        m = len(indices) - 1
        starts.append(column); sizes.append(m)
        h_nonref.extend(h[list(indices[:-1])])
        h_ref.append(h[indices[-1]])
        column += m
    return EntropyPreconditioner(np.asarray(starts), np.asarray(sizes),
                                 np.asarray(h_nonref), np.asarray(h_ref))


@dataclass
class SpectralBound:
    gamma_path: float          # max_p sum_{a in p} t'_a w_a  (one path-wise gather)
    gamma_regional: float      # sum_k gamma_k  (one scalar per authority)
    gamma_by_region: FloatArray


def congestion_gamma(problem, link_region, hdv_path_flow, av_path_flow,
                     n_regions: int) -> SpectralBound:
    from brn.sue.operators import (bpr_link_derivatives, link_flows_from_paths,
                                  sparse_incidence_matrix)
    D = sparse_incidence_matrix(problem).tocsr()
    group_by_id = {g.group_id: g for g in problem.groups}
    theta = np.asarray([group_by_id[p.group_id].cost_sensitivity for p in problem.paths])
    v = link_flows_from_paths(problem, hdv_path_flow, av_path_flow)
    tprime = bpr_link_derivatives(problem, v)
    w = D @ (theta * np.asarray(hdv_path_flow))
    per_link = tprime * w
    total = np.asarray(D.T @ per_link).ravel()
    by_region = np.zeros(n_regions)
    for k in range(n_regions):
        mask = (np.asarray(link_region) == k).astype(float)
        by_region[k] = float(np.max(np.asarray(D.T @ (per_link * mask)).ravel()))
    return SpectralBound(float(total.max()), float(by_region.sum()), by_region)


def sharpened_gamma(problem, link_region, hdv_path_flow, av_path_flow,
                    n_regions: int) -> SpectralBound:
    """Group-cancellation bound.

    For a reduced direction, the group's path changes sum to zero, so on link a
    the paths of group g through a carry exactly minus the paths of g that avoid
    a. Cauchy-Schwarz is applied to the lighter of the two sides (by theta f),
    and groups whose every path uses a drop out. With T_a the union of the
    chosen sides,  y^T C y <= sum_r z_r^2/(theta f_r) sum_{a: r in T_a} t'_a w_a,
    w_a = sum_{r in T_a} theta_r f_r, hence gamma <= max_r sum_{a: r in T_a} t'_a w_a.
    """
    from brn.sue.operators import (bpr_link_derivatives, link_flows_from_paths,
                                  sparse_incidence_matrix)
    from scipy import sparse
    D = sparse_incidence_matrix(problem).tocsc()          # links x paths
    group_by_id = {g.group_id: g for g in problem.groups}
    theta = np.asarray([group_by_id[p.group_id].cost_sensitivity for p in problem.paths])
    weight = theta * np.asarray(hdv_path_flow, dtype=np.float64)
    v = link_flows_from_paths(problem, hdv_path_flow, av_path_flow)
    tprime = bpr_link_derivatives(problem, v)
    n_links = D.shape[0]
    rows, cols = [], []
    for group in problem.groups:
        idx = np.asarray(problem.path_indices(group.group_id), dtype=int)
        sub = D[:, idx].tocsc()
        used = np.unique(sub.indices) if sub.nnz else np.array([], int)
        through = {a: [] for a in used}
        for j, pth in enumerate(idx):
            for a in sub.indices[sub.indptr[j]:sub.indptr[j + 1]]:
                through[a].append(pth)
        for a, members in through.items():
            if len(members) == idx.size:
                continue                                  # cancels exactly
            others = [p for p in idx if p not in members]
            side = members if weight[members].sum() <= weight[others].sum() else others
            rows.extend([a] * len(side)); cols.extend(side)
    T = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=D.shape)
    T.data[:] = 1.0
    w = T @ weight
    per_link = tprime * w
    total = np.asarray(T.T @ per_link).ravel()
    by_region = np.zeros(n_regions)
    region = np.asarray(link_region)
    for k in range(n_regions):
        by_region[k] = float(np.max(np.asarray(T.T @ (per_link * (region == k))).ravel()))
    return SpectralBound(float(total.max()), float(by_region.sum()), by_region)


def heavy_ball_parameters(lower: float, upper: float) -> tuple[float, float, float]:
    sl, su = np.sqrt(lower), np.sqrt(upper)
    alpha = 4.0 / (sl + su) ** 2
    rho = (su - sl) / (su + sl)
    return alpha, rho * rho, rho


def distributed_spectral_solve(operator: DistributedOperator, rhs: FloatArray,
                               preconditioner: EntropyPreconditioner,
                               upper: float, rounds: int,
                               initial: FloatArray | None = None,
                               lower: float = 1.0,
                               method: str = "chebyshev") -> AdjointSolution:
    """Fixed-round, reduction-free solve of A x = b on spec(P^{-1}A) in [lower, upper].

    method='heavyball': stationary second-order Richardson (analyzed in Thm 2).
    method='chebyshev': Chebyshev semi-iteration on the same interval.
    No inner product is formed; the ledger therefore records no scalar bits.
    """
    b = np.asarray(rhs, dtype=np.float64)
    x = np.zeros_like(b) if initial is None else np.array(initial, dtype=np.float64)
    budget = int(rounds)

    def residual(point):
        # A 0 = 0 needs no exchange; every other product is one counted round.
        nonlocal budget
        if not np.any(point):
            return b.copy()
        budget -= 1
        return b - operator.matvec(point)

    if method == "heavyball":
        alpha, beta, _ = heavy_ball_parameters(lower, upper)
        x_prev = x.copy()
        while budget > 0 or not np.any(x):
            r = residual(x)
            x_new = x + alpha * preconditioner.apply(r) + beta * (x - x_prev)
            x_prev, x = x, x_new
            if budget <= 0:
                break
    elif method == "chebyshev":
        centre = 0.5 * (upper + lower)
        half = 0.5 * (upper - lower)
        sigma = centre / half
        rho_k = 1.0 / sigma
        r = residual(x)
        d = preconditioner.apply(r) / centre
        x = x + d
        while budget > 0:
            r = residual(x)
            rho_next = 1.0 / (2.0 * sigma - rho_k)
            d = rho_next * rho_k * d + (2.0 * rho_next / half) * preconditioner.apply(r)
            x = x + d
            rho_k = rho_next
    else:
        raise ValueError(method)
    return AdjointSolution(adjoint=x, iterations=int(rounds), residual=float("nan"),
                           residual_history=(), ledger=operator.ledger,
                           converged=True)
