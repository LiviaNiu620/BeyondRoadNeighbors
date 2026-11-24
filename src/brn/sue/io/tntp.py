"""Strict, deterministic readers for standard TNTP network and trip files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as FilePath
import math
import re

from ..model import DemandGroup, Link, Path, ProblemData
from ..path_generation import deterministic_k_shortest_paths

_ORIGIN = re.compile(r"^\s*Origin\s+(\S+)", re.IGNORECASE)
_DEMAND = re.compile(
    r"([^\s:;]+)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass(frozen=True, slots=True)
class ODDemandRecord:
    origin: str
    destination: str
    demand: float

    def __post_init__(self) -> None:
        demand = float(self.demand)
        if not math.isfinite(demand) or demand < 0.0:
            raise ValueError("TNTP demand must be finite and nonnegative")
        object.__setattr__(self, "demand", demand)


def _data_lines(file_path: str | FilePath) -> tuple[str, ...]:
    path = FilePath(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(path.read_text(encoding="utf-8-sig").splitlines())


def read_tntp_network(file_path: str | FilePath) -> tuple[Link, ...]:
    """Parse TNTP directed links in file order.

    Required columns are initial node, terminal node, capacity, length, free-flow
    time, BPR alpha, and BPR power. Length is parsed for schema validation but is not
    yet stored because the Study 1 core cost model uses time and capacity.
    """

    links: list[Link] = []
    after_metadata = False
    for line_number, raw_line in enumerate(_data_lines(file_path), start=1):
        stripped = raw_line.strip()
        if not after_metadata:
            if stripped.upper().startswith("<END OF METADATA>"):
                after_metadata = True
            continue
        if not stripped or stripped.startswith("~"):
            continue
        fields = stripped.rstrip(";").split()
        if len(fields) < 7:
            raise ValueError(
                f"TNTP network row {line_number} has fewer than seven fields"
            )
        tail, head = fields[0], fields[1]
        try:
            capacity = float(fields[2])
            length = float(fields[3])
            free_flow_time = float(fields[4])
            alpha = float(fields[5])
            beta = float(fields[6])
        except ValueError as error:
            raise ValueError(
                f"TNTP network row {line_number} contains a nonnumeric cost field"
            ) from error
        if not math.isfinite(length) or length < 0.0:
            raise ValueError(f"TNTP network row {line_number} has invalid length")
        links.append(
            Link(
                link_id=f"{len(links)}:{tail}->{head}",
                tail=tail,
                head=head,
                free_flow_time=free_flow_time,
                capacity=capacity,
                alpha=alpha,
                beta=beta,
            )
        )
    if not after_metadata:
        raise ValueError("TNTP network is missing <END OF METADATA>")
    if not links:
        raise ValueError("TNTP network contains no link rows")
    return tuple(links)


def read_tntp_demands(
    file_path: str | FilePath, include_zero: bool = False
) -> tuple[ODDemandRecord, ...]:
    """Parse TNTP OD demand records in origin-block order."""

    records: list[ODDemandRecord] = []
    after_metadata = False
    origin: str | None = None
    for line_number, raw_line in enumerate(_data_lines(file_path), start=1):
        stripped = raw_line.strip()
        if not after_metadata:
            if stripped.upper().startswith("<END OF METADATA>"):
                after_metadata = True
            continue
        match = _ORIGIN.match(raw_line)
        if match:
            origin = match.group(1)
            continue
        if not stripped:
            continue
        if origin is None:
            raise ValueError(
                f"TNTP trips row {line_number} appears before an Origin block"
            )
        for destination, demand_text in _DEMAND.findall(raw_line):
            demand = float(demand_text)
            if include_zero or demand > 0.0:
                records.append(ODDemandRecord(origin, destination, demand))
    if not after_metadata:
        raise ValueError("TNTP trips file is missing <END OF METADATA>")
    if not records and not include_zero:
        raise ValueError("TNTP trips file contains no positive OD demand")
    return tuple(records)


def demand_groups_from_tntp(
    records: tuple[ODDemandRecord, ...],
    av_share: float,
    cost_sensitivity: float = 1.0,
    av_cost_sensitivity: float | None = None,
    recommendation_sensitivity: float = 0.0,
) -> tuple[DemandGroup, ...]:
    """Split TNTP OD totals into one HDV/AV group per positive OD record."""

    av_share = float(av_share)
    if not math.isfinite(av_share) or not 0.0 <= av_share <= 1.0:
        raise ValueError("av_share must lie in [0, 1]")
    groups: list[DemandGroup] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        if record.demand <= 0.0 or record.origin == record.destination:
            continue
        key = (record.origin, record.destination)
        if key in seen:
            raise ValueError(f"duplicate TNTP OD record: {key}")
        seen.add(key)
        groups.append(
            DemandGroup(
                group_id=f"{record.origin}->{record.destination}:all",
                origin=record.origin,
                destination=record.destination,
                hdv_demand=(1.0 - av_share) * record.demand,
                av_demand=av_share * record.demand,
                cost_sensitivity=cost_sensitivity,
                av_cost_sensitivity=av_cost_sensitivity,
                recommendation_sensitivity=recommendation_sensitivity,
            )
        )
    if not groups:
        raise ValueError("no positive non-diagonal OD groups were produced")
    return tuple(groups)


def assemble_tntp_problem(
    network_file: str | FilePath,
    trips_file: str | FilePath,
    av_share: float,
    paths_per_group: int,
    cost_sensitivity: float = 1.0,
    av_cost_sensitivity: float | None = None,
    recommendation_sensitivity: float = 0.0,
) -> ProblemData:
    """Build complete immutable problem data without implicit OD truncation."""

    links = read_tntp_network(network_file)
    records = read_tntp_demands(trips_file)
    groups = demand_groups_from_tntp(
        records,
        av_share=av_share,
        cost_sensitivity=cost_sensitivity,
        av_cost_sensitivity=av_cost_sensitivity,
        recommendation_sensitivity=recommendation_sensitivity,
    )
    paths: list[Path] = []
    od_cache: dict[tuple[str, str], tuple[tuple[int, ...], ...]] = {}
    for group in groups:
        key = (group.origin, group.destination)
        if key not in od_cache:
            od_cache[key] = deterministic_k_shortest_paths(
                links,
                group.origin,
                group.destination,
                k=paths_per_group,
            )
        candidates = od_cache[key]
        if not candidates:
            raise ValueError(f"no path exists for TNTP OD {key}")
        for rank, link_indices in enumerate(candidates):
            paths.append(
                Path(
                    path_id=f"{group.group_id}:p{rank}",
                    group_id=group.group_id,
                    link_indices=link_indices,
                )
            )
    return ProblemData(links=links, groups=groups, paths=tuple(paths))


def assemble_tntp_subset_problem(
    network_file: str | FilePath,
    trips_file: str | FilePath,
    od_pairs: tuple[tuple[str, str], ...],
    av_share: float,
    paths_per_group: int,
    cost_sensitivity: float = 1.0,
    av_cost_sensitivity: float | None = None,
    recommendation_sensitivity: float = 0.0,
) -> ProblemData:
    """Build an explicitly declared OD correctness fixture.

    ``od_pairs`` is ordered, must contain no duplicates, and is never interpreted
    as "the first N" records.  This prevents accidental lexicographic OD
    truncation from being presented as a network-scale experiment.
    """

    requested = tuple((str(origin), str(destination)) for origin, destination in od_pairs)
    if not requested:
        raise ValueError("od_pairs must contain at least one explicit OD pair")
    if len(requested) != len(set(requested)):
        raise ValueError("od_pairs contains a duplicate")
    records_by_od = {
        (record.origin, record.destination): record
        for record in read_tntp_demands(trips_file)
    }
    missing = tuple(pair for pair in requested if pair not in records_by_od)
    if missing:
        raise ValueError(f"requested TNTP OD pairs are absent: {missing}")
    records = tuple(records_by_od[pair] for pair in requested)
    links = read_tntp_network(network_file)
    groups = demand_groups_from_tntp(
        records,
        av_share=av_share,
        cost_sensitivity=cost_sensitivity,
        av_cost_sensitivity=av_cost_sensitivity,
        recommendation_sensitivity=recommendation_sensitivity,
    )
    paths: list[Path] = []
    for group in groups:
        candidates = deterministic_k_shortest_paths(
            links, group.origin, group.destination, k=paths_per_group
        )
        if not candidates:
            raise ValueError(
                f"no path exists for explicitly selected TNTP OD "
                f"{(group.origin, group.destination)}"
            )
        for rank, link_indices in enumerate(candidates):
            paths.append(
                Path(
                    path_id=f"{group.group_id}:p{rank}",
                    group_id=group.group_id,
                    link_indices=link_indices,
                )
            )
    return ProblemData(links=links, groups=groups, paths=tuple(paths))
