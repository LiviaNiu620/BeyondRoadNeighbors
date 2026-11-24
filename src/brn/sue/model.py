"""Immutable topology and demand definitions for Study 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Link:
    """A directed link with a monotone BPR performance function."""

    link_id: str
    tail: str
    head: str
    free_flow_time: float
    capacity: float
    alpha: float = 0.15
    beta: float = 4.0

    def __post_init__(self) -> None:
        if not self.link_id:
            raise ValueError("link_id must be non-empty")
        if not self.tail or not self.head:
            raise ValueError("link endpoints must be non-empty")
        if self.tail == self.head:
            raise ValueError(f"self-loop is not allowed in the core: {self.link_id}")

        fft = _finite(self.free_flow_time, "free_flow_time")
        capacity = _finite(self.capacity, "capacity")
        alpha = _finite(self.alpha, "alpha")
        beta = _finite(self.beta, "beta")
        if fft <= 0.0:
            raise ValueError("free_flow_time must be positive")
        if capacity <= 0.0:
            raise ValueError("capacity must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be nonnegative")
        if beta < 1.0:
            raise ValueError("beta must be at least one in the core model")

        object.__setattr__(self, "free_flow_time", fft)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)


@dataclass(frozen=True, slots=True)
class DemandGroup:
    """One OD/behavioral class with HDV and coordinated AV demand."""

    group_id: str
    origin: str
    destination: str
    hdv_demand: float
    av_demand: float = 0.0
    cost_sensitivity: float = 1.0
    av_cost_sensitivity: float | None = None
    recommendation_sensitivity: float = 0.0

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not self.origin or not self.destination:
            raise ValueError("origin and destination must be non-empty")
        if self.origin == self.destination:
            raise ValueError(f"origin equals destination for group {self.group_id}")

        hdv = _finite(self.hdv_demand, "hdv_demand")
        av = _finite(self.av_demand, "av_demand")
        theta = _finite(self.cost_sensitivity, "cost_sensitivity")
        av_theta = _finite(
            self.cost_sensitivity
            if self.av_cost_sensitivity is None
            else self.av_cost_sensitivity,
            "av_cost_sensitivity",
        )
        delta = _finite(
            self.recommendation_sensitivity, "recommendation_sensitivity"
        )
        if hdv < 0.0 or av < 0.0:
            raise ValueError("demands must be nonnegative")
        if hdv + av <= 0.0:
            raise ValueError("a group must have positive total demand")
        if theta <= 0.0:
            raise ValueError("cost_sensitivity must be positive")
        if av_theta <= 0.0:
            raise ValueError("av_cost_sensitivity must be positive")
        if delta < 0.0:
            raise ValueError("recommendation_sensitivity must be nonnegative")

        object.__setattr__(self, "hdv_demand", hdv)
        object.__setattr__(self, "av_demand", av)
        object.__setattr__(self, "cost_sensitivity", theta)
        object.__setattr__(self, "av_cost_sensitivity", av_theta)
        object.__setattr__(self, "recommendation_sensitivity", delta)


@dataclass(frozen=True, slots=True)
class Path:
    """A path column belonging to exactly one demand group."""

    path_id: str
    group_id: str
    link_indices: tuple[int, ...]
    preference_constant: float = 0.0
    av_preference_constant: float | None = None

    def __post_init__(self) -> None:
        if not self.path_id:
            raise ValueError("path_id must be non-empty")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        links = tuple(int(index) for index in self.link_indices)
        if not links:
            raise ValueError(f"path {self.path_id} has no links")
        if len(set(links)) != len(links):
            raise ValueError(
                f"path {self.path_id} repeats a link; cyclic paths are excluded"
            )
        preference = _finite(self.preference_constant, "preference_constant")
        av_preference = _finite(
            self.preference_constant
            if self.av_preference_constant is None
            else self.av_preference_constant,
            "av_preference_constant",
        )
        object.__setattr__(self, "link_indices", links)
        object.__setattr__(self, "preference_constant", preference)
        object.__setattr__(self, "av_preference_constant", av_preference)


@dataclass(frozen=True, slots=True)
class ProblemData:
    """Validated immutable restricted-path problem data."""

    links: tuple[Link, ...]
    groups: tuple[DemandGroup, ...]
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        links = tuple(self.links)
        groups = tuple(self.groups)
        paths = tuple(self.paths)
        if not links:
            raise ValueError("problem must contain at least one link")
        if not groups:
            raise ValueError("problem must contain at least one demand group")
        if not paths:
            raise ValueError("problem must contain at least one path")

        self._require_unique((link.link_id for link in links), "link_id")
        self._require_unique((group.group_id for group in groups), "group_id")
        self._require_unique((path.path_id for path in paths), "path_id")

        group_ids = {group.group_id for group in groups}
        groups_by_id = {group.group_id: group for group in groups}
        path_count = {group_id: 0 for group_id in group_ids}
        for path in paths:
            if path.group_id not in group_ids:
                raise ValueError(
                    f"path {path.path_id} references unknown group {path.group_id}"
                )
            if min(path.link_indices) < 0 or max(path.link_indices) >= len(links):
                raise ValueError(f"path {path.path_id} has an invalid link index")
            group = groups_by_id[path.group_id]
            path_links = tuple(links[index] for index in path.link_indices)
            if path_links[0].tail != group.origin:
                raise ValueError(
                    f"path {path.path_id} does not start at {group.origin}"
                )
            if path_links[-1].head != group.destination:
                raise ValueError(
                    f"path {path.path_id} does not end at {group.destination}"
                )
            for first, second in zip(path_links, path_links[1:]):
                if first.head != second.tail:
                    raise ValueError(f"path {path.path_id} is not link-continuous")
            path_count[path.group_id] += 1

        for group in groups:
            if path_count[group.group_id] == 0:
                raise ValueError(f"group {group.group_id} has no path")

        object.__setattr__(self, "links", links)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "paths", paths)

    @staticmethod
    def _require_unique(values, name: str) -> None:
        values = tuple(values)
        if len(values) != len(set(values)):
            raise ValueError(f"{name} values must be unique")

    @property
    def number_of_links(self) -> int:
        return len(self.links)

    @property
    def number_of_paths(self) -> int:
        return len(self.paths)

    def group(self, group_id: str) -> DemandGroup:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)

    def path_indices(self, group_id: str) -> tuple[int, ...]:
        indices = tuple(
            index for index, path in enumerate(self.paths) if path.group_id == group_id
        )
        if not indices:
            raise KeyError(group_id)
        return indices

    @property
    def total_hdv_demand(self) -> float:
        return sum(group.hdv_demand for group in self.groups)

    @property
    def total_av_demand(self) -> float:
        return sum(group.av_demand for group in self.groups)


def scaled_demand_problem(problem: ProblemData, factor: float) -> ProblemData:
    """Return the same topology/path library with every demand scaled uniformly."""

    factor = _finite(factor, "demand_scale")
    if factor <= 0.0:
        raise ValueError("demand_scale must be positive")
    if factor == 1.0:
        return problem
    groups = tuple(
        replace(
            group,
            hdv_demand=factor * group.hdv_demand,
            av_demand=factor * group.av_demand,
        )
        for group in problem.groups
    )
    return ProblemData(links=problem.links, groups=groups, paths=problem.paths)
