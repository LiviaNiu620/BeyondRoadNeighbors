"""Spatial partition of the road network and the graphs it induces.

An agent owns one region of links. It owns the demand groups whose origin lies in
its region, and therefore the reduced coordinates of those groups.

Two graphs on the agent set matter and are NOT the same:

* ``road_adjacency``  -- regions joined by a link. The conventional choice.
* ``coupling``        -- agent k needs a message from agent l iff some reduced
  coordinate owned by k touches a link in region l. This is what the
  implicit-gradient computation actually requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int_]


def road_relay_multiplier(decomposition: "RegionDecomposition") -> float:
    """Mean shortest-path hops needed to carry induced coupling over road adjacency.

    This is a forwarding-cost multiplier for a relay *emulation*. It does not
    implement asynchronous routing, contention, or failures; disconnected
    coupling pairs return infinity and must be treated as unavailable.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(decomposition.n_regions))
    graph.add_edges_from(decomposition.road_adjacency)
    distances = []
    for a, b in decomposition.coupling_undirected:
        try:
            distances.append(float(nx.shortest_path_length(graph, a, b)))
        except nx.NetworkXNoPath:
            return float("inf")
    return float(np.mean(distances)) if distances else 1.0


@dataclass(frozen=True)
class RegionDecomposition:
    """Everything the decentralized solver needs to know about the partition."""

    n_regions: int
    link_region: IntArray            # link index -> region
    path_owner: IntArray             # path index -> owning agent
    coordinate_owner: IntArray       # reduced coordinate -> owning agent
    coordinate_touch: tuple[frozenset[int], ...]  # reduced coordinate -> regions it touches
    road_adjacency: frozenset[tuple[int, int]]    # undirected, sorted pairs
    coupling: frozenset[tuple[int, int]]          # DIRECTED (receiver, sender)

    @property
    def coupling_undirected(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (min(a, b), max(a, b)) for a, b in self.coupling if a != b
        )

    def owned_coordinates(self, region: int) -> IntArray:
        return np.where(self.coordinate_owner == region)[0]

    def senders_for(self, region: int) -> tuple[int, ...]:
        """Agents whose contribution agent ``region`` needs (excluding itself)."""
        return tuple(
            sorted(sender for receiver, sender in self.coupling
                   if receiver == region and sender != region)
        )

    def summary(self) -> dict[str, float | int]:
        undirected = self.coupling_undirected
        return {
            "n_regions": self.n_regions,
            "road_adjacency_edges": len(self.road_adjacency),
            "coupling_edges": len(undirected),
            "coupled_not_adjacent": len(undirected - self.road_adjacency),
            "mean_senders": float(
                np.mean([len(self.senders_for(k)) for k in range(self.n_regions)])
            ),
        }


def node_block_partition(problem, n_regions: int) -> dict[str, int]:
    """Contiguous node-id blocks. Deterministic placeholder for a real partitioner."""
    nodes = sorted(
        {l.tail for l in problem.links} | {l.head for l in problem.links},
        key=lambda s: int(s),
    )
    assignment: dict[str, int] = {}
    for region, block in enumerate(np.array_split(np.arange(len(nodes)), n_regions)):
        for index in block:
            assignment[nodes[index]] = region
    return assignment


def decompose(problem, coordinates, node_region: dict[str, int],
              n_regions: int) -> RegionDecomposition:
    """Build the decomposition and both graphs from a node->region assignment."""

    link_region = np.array([node_region[link.tail] for link in problem.links])

    group_by_id = {group.group_id: group for group in problem.groups}
    path_owner = np.array(
        [node_region[group_by_id[path.group_id].origin] for path in problem.paths]
    )

    # Reconstruct the (nonreference, reference) pair behind every reduced
    # coordinate WITHOUT materializing the dense basis: the conservation basis is
    # built group by group, one column per nonreference path, reference last.
    pairs: list[tuple[int, int]] = []
    for group in problem.groups:
        indices = problem.path_indices(group.group_id)
        reference = indices[-1]
        for path_index in indices[:-1]:
            pairs.append((int(path_index), int(reference)))

    expected = tuple(int(i) for i, _ in pairs)
    if expected != tuple(int(i) for i in coordinates.nonreference_indices):
        raise ValueError(
            "reduced-coordinate order does not match the upstream basis; "
            "the decomposition cannot be trusted"
        )
    n_coordinates = len(pairs)

    path_regions = [
        frozenset(int(link_region[i]) for i in path.link_indices)
        for path in problem.paths
    ]

    coordinate_owner = np.empty(n_coordinates, dtype=int)
    touch: list[frozenset[int]] = []
    for index, (nonreference, reference) in enumerate(pairs):
        owners = {int(path_owner[nonreference]), int(path_owner[reference])}
        if len(owners) != 1:
            raise ValueError(
                f"reduced coordinate {index} spans several owners {owners}; "
                "the conservation basis must stay inside one demand group"
            )
        owner = owners.pop()
        coordinate_owner[index] = owner
        regions = set(path_regions[nonreference]) | set(path_regions[reference])
        regions.add(owner)          # the entropy term is local to the owner
        touch.append(frozenset(regions))

    road_adjacency = set()
    for link in problem.links:
        a, b = node_region[link.tail], node_region[link.head]
        if a != b:
            road_adjacency.add((min(a, b), max(a, b)))

    coupling = set()
    for index in range(n_coordinates):
        receiver = int(coordinate_owner[index])
        for sender in touch[index]:
            coupling.add((receiver, int(sender)))

    return RegionDecomposition(
        n_regions=n_regions,
        link_region=link_region,
        path_owner=path_owner,
        coordinate_owner=coordinate_owner,
        coordinate_touch=tuple(touch),
        road_adjacency=frozenset(road_adjacency),
        coupling=frozenset(coupling),
    )


def communication_graph(decomposition: RegionDecomposition, topology: str
                        ) -> frozenset[tuple[int, int]]:
    """Directed (receiver, sender) pairs the agents are allowed to use.

    ``full``      all-to-all
    ``coupling``  exactly what the gradient needs (the proposed topology)
    ``road``      road adjacency only (the conventional, insufficient choice)
    ``none``      independent learners
    """
    k = decomposition.n_regions
    self_edges = {(i, i) for i in range(k)}
    if topology == "full":
        return frozenset(self_edges | {(i, j) for i in range(k) for j in range(k)})
    if topology == "coupling":
        return frozenset(self_edges | set(decomposition.coupling))
    if topology == "road":
        road = set()
        for a, b in decomposition.road_adjacency:
            road.add((a, b))
            road.add((b, a))
        return frozenset(self_edges | road)
    if topology == "none":
        return frozenset(self_edges)
    if topology.startswith("road") and topology[4:].isdigit():
        # fixed kappa-hop mask over road adjacency, e.g. "road2".
        # Each authority hears every region within kappa road hops.
        import networkx as nx
        kappa = int(topology[4:])
        graph = nx.Graph()
        graph.add_nodes_from(range(k))
        graph.add_edges_from(decomposition.road_adjacency)
        pairs = set(self_edges)
        for i in range(k):
            reach = nx.single_source_shortest_path_length(graph, i, cutoff=kappa)
            pairs |= {(i, j) for j in reach}
        return frozenset(pairs)
    raise ValueError(f"unknown topology {topology!r}")


# --------------------------------------------------------------------------
# Spectral partitioning
# --------------------------------------------------------------------------

def _fiedler_split(adjacency, nodes: list[int]) -> tuple[list[int], list[int]]:
    """Balanced bisection by the Fiedler vector of the induced subgraph."""
    import numpy as _np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    index = {node: i for i, node in enumerate(nodes)}
    rows, cols = [], []
    for node in nodes:
        for neighbour in adjacency[node]:
            if neighbour in index:
                rows.append(index[node])
                cols.append(index[neighbour])
    size = len(nodes)
    if not rows:
        half = size // 2
        return nodes[:half], nodes[half:]
    weights = _np.ones(len(rows))
    affinity = csr_matrix((weights, (rows, cols)), shape=(size, size))
    affinity = ((affinity + affinity.T) > 0).astype(float)
    degree = _np.asarray(affinity.sum(axis=1)).ravel()
    laplacian = csr_matrix((degree, (range(size), range(size))), shape=(size, size))
    laplacian = laplacian - affinity
    try:
        # Fixed starting vector: ARPACK otherwise starts from a random vector,
        # which made the partition -- and therefore every per-jurisdiction
        # result -- change from run to run.
        start = _np.random.default_rng(12345).random(size)
        _, vectors = eigsh(laplacian, k=2, sigma=-1e-6, which="LM", v0=start)
        fiedler = vectors[:, 1]
        # Eigenvectors are defined up to sign; pin it so the split is canonical.
        anchor = int(_np.argmax(_np.abs(fiedler)))
        if fiedler[anchor] < 0:
            fiedler = -fiedler
    except Exception:                       # disconnected or tiny: fall back
        fiedler = _np.asarray(range(size), dtype=float)
    order = _np.argsort(fiedler, kind="stable")
    half = size // 2
    left = [nodes[i] for i in order[:half]]
    right = [nodes[i] for i in order[half:]]
    return left, right


def spectral_partition(problem, n_regions: int) -> dict[str, int]:
    """Recursive balanced spectral bisection of the undirected road graph.

    A principled stand-in for METIS that needs no extra dependency: it minimizes
    the edge cut while keeping the regions balanced, which is exactly the
    conventional districting a traffic agency would produce.
    """
    if n_regions < 1:
        raise ValueError("n_regions must be positive")

    node_ids = sorted(
        {l.tail for l in problem.links} | {l.head for l in problem.links},
        key=lambda s: int(s),
    )
    position = {node: i for i, node in enumerate(node_ids)}
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(node_ids))}
    for link in problem.links:
        a, b = position[link.tail], position[link.head]
        adjacency[a].add(b)
        adjacency[b].add(a)

    parts: list[list[int]] = [list(range(len(node_ids)))]
    while len(parts) < n_regions:
        parts.sort(key=len, reverse=True)
        largest = parts.pop(0)
        if len(largest) < 2:
            parts.append(largest)
            break
        left, right = _fiedler_split(adjacency, largest)
        parts.extend([left, right])

    # Canonical labels: region 0 contains the lowest-numbered node, and so on.
    parts.sort(key=lambda part: min(part) if part else 10**9)
    assignment: dict[str, int] = {}
    for region, part in enumerate(parts):
        for i in part:
            assignment[node_ids[i]] = region
    for node in node_ids:                   # safety: never leave a node unassigned
        assignment.setdefault(node, 0)
    return assignment


def build_partition(problem, n_regions: int, method: str = "spectral") -> dict[str, int]:
    if method == "spectral":
        return spectral_partition(problem, n_regions)
    if method == "blocks":
        return node_block_partition(problem, n_regions)
    raise ValueError(f"unknown partition method {method!r}")
