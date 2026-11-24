"""Per-region assembly of the reduced KKT system, sparse.

The exact decomposition (verified to machine precision by
``scripts/verify_partition_decomposition.py``):

    F_y = sum_k E^T ( D_k^T T'_k D_k + H_e^(k) ) E
    J_y = sum_k E^T D_k^T m_k

where ``D_k`` keeps the rows of the link-path incidence belonging to region k,
``T'_k`` the BPR derivatives on those links, ``H_e^(k)`` the entropy diagonal of
the paths that region k owns, and ``m`` the marginal link cost.

Agent k assembles its own block from its own link observations. Nothing here
communicates. Everything is scipy.sparse: a full-library Anaheim instance has
thousands of reduced coordinates and dense blocks would not fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalSystem:
    """One agent's share of the reduced KKT system."""

    region: int
    matrix: sparse.csr_matrix       # F_y^(k) = E^T (D_k^T T'_k D_k + H_e^(k)) E
    congestion: sparse.csr_matrix   # F_a^(k) = E^T D_k^T T'_k D_k E (no entropy term)
    rhs: FloatArray                 # J_y^(k)
    n_links: int
    nnz: int


def assemble_local_systems(problem, coordinates, decomposition,
                           hdv_path_flow, av_path_flow) -> list[LocalSystem]:
    from brn.sue.operators import (
        bpr_link_costs,
        bpr_link_derivatives,
        link_flows_from_paths,
        sparse_incidence_matrix,
    )
    from brn.sue.sparse_sensitivity import sparse_reduced_basis

    incidence = sparse_incidence_matrix(problem).tocsr()      # links x paths
    basis = sparse_reduced_basis(coordinates).tocsc()         # paths x coordinates

    link_flow = link_flows_from_paths(problem, hdv_path_flow, av_path_flow)
    link_derivative = bpr_link_derivatives(problem, link_flow)
    marginal_cost = bpr_link_costs(problem, link_flow) + link_flow * link_derivative

    group_by_id = {group.group_id: group for group in problem.groups}
    theta = np.asarray(
        [group_by_id[p.group_id].cost_sensitivity for p in problem.paths]
    )
    hdv = np.asarray(hdv_path_flow, dtype=np.float64)
    if np.any(hdv <= 0.0):
        raise ValueError("assembly requires strictly positive HDV path flow")

    n_paths = problem.number_of_paths
    systems: list[LocalSystem] = []
    for region in range(decomposition.n_regions):
        local_links = np.where(decomposition.link_region == region)[0]
        local_incidence = incidence[local_links, :]
        scaled = sparse.diags(link_derivative[local_links]) @ local_incidence
        congestion = (local_incidence.T @ scaled).tocsr()

        entropy_diagonal = np.zeros(n_paths)
        owned = np.where(decomposition.path_owner == region)[0]
        entropy_diagonal[owned] = 1.0 / (theta[owned] * hdv[owned])
        full = congestion + sparse.diags(entropy_diagonal)

        matrix = (basis.T @ full @ basis).tocsr()
        matrix.eliminate_zeros()
        congestion_reduced = (basis.T @ congestion @ basis).tocsr()
        congestion_reduced.eliminate_zeros()
        rhs = np.asarray(
            basis.T @ (local_incidence.T @ marginal_cost[local_links])
        ).ravel()
        systems.append(
            LocalSystem(
                region=region,
                matrix=matrix,
                congestion=congestion_reduced,
                rhs=rhs,
                n_links=int(local_links.size),
                nnz=int(matrix.nnz),
            )
        )
    return systems


def centralized_system(systems: list[LocalSystem]):
    """The exact global system, for reference only. Never used by the agents."""
    matrix = systems[0].matrix.copy()
    for system in systems[1:]:
        matrix = matrix + system.matrix
    rhs = sum(system.rhs for system in systems)
    return matrix.tocsr(), rhs
