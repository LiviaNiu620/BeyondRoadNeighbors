"""Feasible reference controls.

The experiments evaluate the adjoint at a declared, reproducible control point.
Filling the reduced AV coordinates with a constant is NOT feasible on a real
network: small-demand groups end up with a negative reference-path flow. The
even split below is feasible for every group by construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def uniform_av_reduced(problem, coordinates) -> FloatArray:
    """Split each group's AV demand evenly across its paths."""
    values: list[float] = []
    for group in problem.groups:
        indices = problem.path_indices(group.group_id)
        share = group.av_demand / len(indices)
        values.extend([share] * (len(indices) - 1))
    reduced = np.asarray(values, dtype=np.float64)
    expected = len(coordinates.nonreference_indices)
    if reduced.size != expected:
        raise ValueError(
            f"reduced AV vector has {reduced.size} entries, expected {expected}"
        )
    return reduced


def uniform_logits(coordinates) -> FloatArray:
    """Zero reference-normalized logits: the uniform recommendation."""
    return np.zeros(len(coordinates.nonreference_indices))


def reference_controls(problem, coordinates) -> tuple[FloatArray, FloatArray]:
    return uniform_av_reduced(problem, coordinates), uniform_logits(coordinates)
