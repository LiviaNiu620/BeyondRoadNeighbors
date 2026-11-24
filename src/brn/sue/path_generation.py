"""Deterministic shortest-path and Yen k-shortest-path construction."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import Link

FloatArray = NDArray[np.float64]


class RelevantNegativeCycleError(RuntimeError):
    """A negative cycle is reachable from origin and can reach destination."""


class PathEnumerationLimitError(RuntimeError):
    """Exact simple-path enumeration exceeded its declared safety limit."""


def _weights(links: tuple[Link, ...], values: ArrayLike | None) -> FloatArray:
    weights = np.asarray(
        [link.free_flow_time for link in links] if values is None else values,
        dtype=np.float64,
    )
    if weights.shape != (len(links),):
        raise ValueError(f"weights must have shape ({len(links)},)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("path weights must be finite and nonnegative")
    return weights


def path_nodes(links: tuple[Link, ...], link_indices: Iterable[int]) -> tuple[str, ...]:
    indices = tuple(link_indices)
    if not indices:
        raise ValueError("a path must contain at least one link")
    selected = tuple(links[index] for index in indices)
    for first, second in zip(selected, selected[1:]):
        if first.head != second.tail:
            raise ValueError("link sequence is not continuous")
    return (selected[0].tail,) + tuple(link.head for link in selected)


def path_weight(link_indices: Iterable[int], weights: ArrayLike) -> float:
    indices = tuple(link_indices)
    vector = np.asarray(weights, dtype=np.float64)
    return float(np.sum(vector[np.asarray(indices, dtype=int)]))


def deterministic_shortest_path(
    links: tuple[Link, ...],
    origin: str,
    destination: str,
    weights: ArrayLike | None = None,
    banned_links: frozenset[int] = frozenset(),
    banned_nodes: frozenset[str] = frozenset(),
) -> tuple[int, ...] | None:
    """Return a deterministic simple shortest path or ``None`` if disconnected."""

    if origin == destination:
        raise ValueError("origin and destination must differ")
    values = _weights(links, weights)
    adjacency: dict[str, list[int]] = {}
    for index, link in enumerate(links):
        adjacency.setdefault(link.tail, []).append(index)
    for outgoing in adjacency.values():
        outgoing.sort(key=lambda index: (links[index].head, index))

    # Cost then link-index signature yields deterministic tie breaking.
    queue: list[tuple[float, tuple[int, ...], str, tuple[str, ...]]] = [
        (0.0, (), origin, (origin,))
    ]
    best: dict[str, tuple[float, tuple[int, ...]]] = {}
    while queue:
        cost, signature, node, visited_nodes = heapq.heappop(queue)
        previous = best.get(node)
        if previous is not None and previous <= (cost, signature):
            continue
        best[node] = (cost, signature)
        if node == destination:
            return signature
        for link_index in adjacency.get(node, ()):
            if link_index in banned_links:
                continue
            link = links[link_index]
            if link.head in banned_nodes or link.head in visited_nodes:
                continue
            new_cost = cost + float(values[link_index])
            if not math.isfinite(new_cost):
                continue
            heapq.heappush(
                queue,
                (
                    new_cost,
                    signature + (link_index,),
                    link.head,
                    visited_nodes + (link.head,),
                ),
            )
    return None


def deterministic_k_shortest_paths(
    links: tuple[Link, ...],
    origin: str,
    destination: str,
    k: int,
    weights: ArrayLike | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return up to ``k`` loopless paths using deterministic Yen enumeration."""

    if k < 1:
        raise ValueError("k must be positive")
    values = _weights(links, weights)
    first = deterministic_shortest_path(links, origin, destination, values)
    if first is None:
        return ()
    accepted: list[tuple[int, ...]] = [first]
    accepted_set = {first}
    candidates: list[tuple[float, tuple[int, ...]]] = []
    candidate_set: set[tuple[int, ...]] = set()

    while len(accepted) < k:
        previous = accepted[-1]
        previous_nodes = path_nodes(links, previous)
        for spur_index in range(len(previous)):
            root_links = previous[:spur_index]
            root_nodes = previous_nodes[: spur_index + 1]
            spur_node = root_nodes[-1]
            removed_links: set[int] = set()
            for path in accepted:
                if path[:spur_index] == root_links and len(path) > spur_index:
                    removed_links.add(path[spur_index])
            spur_path = deterministic_shortest_path(
                links,
                spur_node,
                destination,
                values,
                banned_links=frozenset(removed_links),
                banned_nodes=frozenset(root_nodes[:-1]),
            )
            if spur_path is None:
                continue
            candidate = root_links + spur_path
            if candidate in accepted_set or candidate in candidate_set:
                continue
            candidate_set.add(candidate)
            heapq.heappush(candidates, (path_weight(candidate, values), candidate))

        if not candidates:
            break
        _, next_path = heapq.heappop(candidates)
        candidate_set.remove(next_path)
        accepted.append(next_path)
        accepted_set.add(next_path)

    return tuple(accepted)


def deterministic_signed_shortest_path(
    links: tuple[Link, ...],
    origin: str,
    destination: str,
    weights: ArrayLike,
    banned_links: frozenset[int] = frozenset(),
    banned_nodes: frozenset[str] = frozenset(),
) -> tuple[int, ...] | None:
    """Deterministic Bellman--Ford path for signed weights without relevant cycles."""

    if origin == destination:
        raise ValueError("origin and destination must differ")
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(links),) or not np.all(np.isfinite(values)):
        raise ValueError("signed path weights must be finite and match links")
    nodes = sorted(
        {link.tail for link in links} | {link.head for link in links} | {origin, destination}
    )
    allowed_edges = tuple(
        index
        for index, link in enumerate(links)
        if index not in banned_links
        and link.tail not in banned_nodes
        and link.head not in banned_nodes
    )
    best: dict[str, tuple[float, tuple[int, ...], tuple[str, ...]]] = {
        origin: (0.0, (), (origin,))
    }
    tolerance = 1e-14
    for _ in range(max(len(nodes) - 1, 0)):
        updated = dict(best)
        changed = False
        for edge in allowed_edges:
            link = links[edge]
            prefix = best.get(link.tail)
            if prefix is None or link.head in prefix[2]:
                continue
            candidate = (
                prefix[0] + float(values[edge]),
                prefix[1] + (edge,),
                prefix[2] + (link.head,),
            )
            incumbent = updated.get(link.head)
            if incumbent is None or candidate[0] < incumbent[0] - tolerance or (
                abs(candidate[0] - incumbent[0]) <= tolerance
                and candidate[1] < incumbent[1]
            ):
                updated[link.head] = candidate
                changed = True
        best = updated
        if not changed:
            break

    # Standard distance-only relaxation detects a relevant negative cycle.  The
    # elementary shortest-path problem is then distinct and is not certified here.
    distance = {node: float("inf") for node in nodes}
    distance[origin] = 0.0
    for _ in range(max(len(nodes) - 1, 0)):
        changed = False
        for edge in allowed_edges:
            link = links[edge]
            candidate = distance[link.tail] + float(values[edge])
            if candidate < distance[link.head] - tolerance:
                distance[link.head] = candidate
                changed = True
        if not changed:
            break
    reverse: dict[str, list[str]] = {}
    for edge in allowed_edges:
        link = links[edge]
        reverse.setdefault(link.head, []).append(link.tail)
    can_reach = {destination}
    stack = [destination]
    while stack:
        node = stack.pop()
        for predecessor in reverse.get(node, ()):
            if predecessor not in can_reach:
                can_reach.add(predecessor)
                stack.append(predecessor)
    for edge in allowed_edges:
        link = links[edge]
        if (
            distance[link.tail] < float("inf")
            and link.head in can_reach
            and distance[link.tail] + float(values[edge])
            < distance[link.head] - tolerance
        ):
            raise RelevantNegativeCycleError(
                "signed pricing graph contains an origin-destination relevant "
                "negative cycle"
            )
    result = best.get(destination)
    return None if result is None else result[1]


def deterministic_signed_k_shortest_paths(
    links: tuple[Link, ...],
    origin: str,
    destination: str,
    k: int,
    weights: ArrayLike,
) -> tuple[tuple[int, ...], ...]:
    """Deterministic Yen enumeration using the certified signed path oracle."""

    if k < 1:
        raise ValueError("k must be positive")
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(links),) or not np.all(np.isfinite(values)):
        raise ValueError("signed path weights must be finite and match links")
    first = deterministic_signed_shortest_path(
        links, origin, destination, values
    )
    if first is None:
        return ()
    accepted = [first]
    accepted_set = {first}
    candidates: list[tuple[float, tuple[int, ...]]] = []
    candidate_set: set[tuple[int, ...]] = set()
    while len(accepted) < k:
        previous = accepted[-1]
        previous_nodes = path_nodes(links, previous)
        for spur_index in range(len(previous)):
            root_links = previous[:spur_index]
            root_nodes = previous_nodes[: spur_index + 1]
            removed_links = {
                path[spur_index]
                for path in accepted
                if path[:spur_index] == root_links and len(path) > spur_index
            }
            spur = deterministic_signed_shortest_path(
                links,
                root_nodes[-1],
                destination,
                values,
                banned_links=frozenset(removed_links),
                banned_nodes=frozenset(root_nodes[:-1]),
            )
            if spur is None:
                continue
            candidate = root_links + spur
            if candidate in accepted_set or candidate in candidate_set:
                continue
            candidate_set.add(candidate)
            heapq.heappush(candidates, (path_weight(candidate, values), candidate))
        if not candidates:
            break
        _, selected = heapq.heappop(candidates)
        candidate_set.remove(selected)
        accepted.append(selected)
        accepted_set.add(selected)
    return tuple(accepted)


def deterministic_all_simple_paths(
    links: tuple[Link, ...],
    origin: str,
    destination: str,
    weights: ArrayLike | None = None,
    maximum_paths: int = 100_000,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every loopless OD path for exact small-network references."""

    if origin == destination:
        raise ValueError("origin and destination must differ")
    if maximum_paths < 1:
        raise ValueError("maximum_paths must be positive")
    values = _weights(links, weights)
    adjacency: dict[str, list[int]] = {}
    for index, link in enumerate(links):
        adjacency.setdefault(link.tail, []).append(index)
    for outgoing in adjacency.values():
        outgoing.sort()
    paths: list[tuple[int, ...]] = []

    def visit(node: str, signature: tuple[int, ...], visited: frozenset[str]) -> None:
        if node == destination:
            paths.append(signature)
            if len(paths) > maximum_paths:
                raise PathEnumerationLimitError(
                    f"more than {maximum_paths} simple paths for {origin}->{destination}"
                )
            return
        for edge in adjacency.get(node, ()):
            head = links[edge].head
            if head in visited:
                continue
            visit(head, signature + (edge,), visited | {head})

    visit(origin, (), frozenset({origin}))
    paths.sort(key=lambda path: (path_weight(path, values), path))
    return tuple(paths)
