"""Heterogeneous regional objectives, and the coalition structures over them.

Each region is an autonomous authority that cares about congestion on ITS OWN
streets:

    J_k(z) = sum_{a in A_k} v_a(z) t_a(v_a(z)),      sum_k J_k = TSTT.

This is the standard not-in-my-backyard externality made precise. Region k can
lower J_k by pushing flow onto region l's links, which raises J_l; the human
population re-equilibrates globally, so every agent's action reaches every other
agent through the follower.

Two structural facts drive the paper:

1. **The objective localizes; the coupling does not.** The gradient of J_k with
   respect to path flows is E^T D_k^T m_k -- exactly the local right-hand side
   agent k already assembles, needing no communication at all. But its adjoint
   still solves F_y lambda^(k) = J_y^(k) with the GLOBAL F_y. Making agents
   selfish removes the need to agree on an objective; it does not remove the
   need to communicate.

2. **Selfishness costs more to compute, not less.** The grand coalition needs one
   adjoint; K singletons need K. Block CG shares the rounds, so the price is
   payload rather than latency -- but it is a price, and it grows with how
   fragmented the authority is.

A coalition structure is a partition of the regions. The grand coalition
recovers the system-wide objective; singletons define the noncooperative game.
Finite gradient-play iterates are not certified optima or Nash equilibria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CoalitionStructure:
    """A partition of the agent set. ``members[c]`` are the regions in coalition c."""

    members: tuple[tuple[int, ...], ...]
    label: str

    @property
    def n_coalitions(self) -> int:
        return len(self.members)

    def coalition_of(self, region: int) -> int:
        for index, group in enumerate(self.members):
            if region in group:
                return index
        raise ValueError(f"region {region} is in no coalition")

    def region_to_coalition(self, n_regions: int) -> NDArray[np.int_]:
        return np.array([self.coalition_of(k) for k in range(n_regions)])


def grand_coalition(n_regions: int) -> CoalitionStructure:
    return CoalitionStructure((tuple(range(n_regions)),), "grand")


def singletons(n_regions: int) -> CoalitionStructure:
    return CoalitionStructure(tuple((k,) for k in range(n_regions)), "singleton")


def blocks(n_regions: int, size: int) -> CoalitionStructure:
    if size < 1:
        raise ValueError("coalition size must be positive")
    members = tuple(
        tuple(range(start, min(start + size, n_regions)))
        for start in range(0, n_regions, size)
    )
    return CoalitionStructure(members, f"blocks{size}")


def build_structure(n_regions: int, spec: str) -> CoalitionStructure:
    """``grand`` | ``singleton`` | ``blocksN``."""
    if spec == "grand":
        return grand_coalition(n_regions)
    if spec == "singleton":
        return singletons(n_regions)
    if spec.startswith("blocks"):
        return blocks(n_regions, int(spec[len("blocks"):]))
    raise ValueError(f"unknown coalition spec {spec!r}")


def coalition_right_hand_sides(systems, structure: CoalitionStructure) -> FloatArray:
    """Column c is the reduced derivative of coalition c's objective.

    Every column is assembled from the links its own members own, so no
    communication happens here -- the aggregation inside a coalition is between
    members who have already agreed to share, and across coalitions there is
    none.
    """
    size = systems[0].rhs.size
    block = np.zeros((size, structure.n_coalitions))
    for index, group in enumerate(structure.members):
        for region in group:
            block[:, index] += systems[region].rhs
    return block


def regional_road_objectives(problem, link_region, hdv_flow, av_flow,
                             n_regions: int) -> FloatArray:
    """Sum v_a*t_a on each region's links, irrespective of travelers' origins."""
    from brn.sue.operators import bpr_link_costs, link_flows_from_paths

    link_flow = link_flows_from_paths(problem, hdv_flow, av_flow)
    contribution = link_flow * bpr_link_costs(problem, link_flow)
    return np.array([
        float(contribution[np.where(link_region == k)[0]].sum())
        for k in range(n_regions)
    ])


# Backward-compatible name; always the road/link objective, never residents.
regional_objectives = regional_road_objectives
