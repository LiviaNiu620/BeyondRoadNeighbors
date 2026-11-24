"""Explicit numerical solution state, separated from immutable topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ProblemData

FloatArray = NDArray[np.float64]


def _readonly_vector(values: ArrayLike, expected: int, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class AssignmentState:
    """A self-contained, read-only assignment snapshot."""

    problem: ProblemData
    hdv_path_flow: FloatArray
    av_path_flow: FloatArray
    recommendation: FloatArray
    link_flow: FloatArray
    link_cost: FloatArray
    path_cost: FloatArray

    def __post_init__(self) -> None:
        number_of_paths = self.problem.number_of_paths
        number_of_links = self.problem.number_of_links
        object.__setattr__(
            self,
            "hdv_path_flow",
            _readonly_vector(self.hdv_path_flow, number_of_paths, "hdv_path_flow"),
        )
        object.__setattr__(
            self,
            "av_path_flow",
            _readonly_vector(self.av_path_flow, number_of_paths, "av_path_flow"),
        )
        object.__setattr__(
            self,
            "recommendation",
            _readonly_vector(self.recommendation, number_of_paths, "recommendation"),
        )
        object.__setattr__(
            self,
            "link_flow",
            _readonly_vector(self.link_flow, number_of_links, "link_flow"),
        )
        object.__setattr__(
            self,
            "link_cost",
            _readonly_vector(self.link_cost, number_of_links, "link_cost"),
        )
        object.__setattr__(
            self,
            "path_cost",
            _readonly_vector(self.path_cost, number_of_paths, "path_cost"),
        )
