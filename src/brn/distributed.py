"""Distributed, compressed, truncated conjugate-gradient solve of the adjoint.

The adjoint system ``F_y lambda = J_y`` is symmetric positive definite and its
matrix is a SUM of locally held matrices, so one CG iteration needs exactly two
exchanges:

* **scatter** -- agent l needs the search direction on every coordinate its own
  matrix touches; the owners of those coordinates send them;
* **gather**  -- the owner of a coordinate needs every other agent's contribution
  to the matrix-vector product on that coordinate.

Agent k must therefore hear from agent l exactly when some coordinate k owns
touches a link in region l -- that is, along the coupling graph. If the
communication graph omits such an edge the contribution is silently dropped and
the matrix-vector product is biased: this is the mechanism the paper measures.

CG scalars are global inner products of local partial sums. They are scalars, so
their cost is negligible next to the vector exchanges; they are counted
separately and transmitted exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .compression import Compressor

FloatArray = NDArray[np.float64]


@dataclass
class CommunicationLedger:
    vector_bits: float = 0.0
    scalar_bits: float = 0.0
    messages: int = 0
    rounds: int = 0
    vector_bit_multiplier: float = 1.0
    per_agent_send_bits: dict[int, float] = field(default_factory=dict)
    per_agent_receive_bits: dict[int, float] = field(default_factory=dict)
    # revision: raw payload per directed pair (headers included, no multiplier), so that
    # forwarding over a routing graph can be charged hop by hop after the run
    pair_bits: dict[tuple[int, int], float] = field(default_factory=dict)

    def record_vector_bits(self, bits: float) -> None:
        """Count payload bits, including optional forwarding overhead."""
        self.vector_bits += float(bits) * self.vector_bit_multiplier

    def record_agent_bits(self, sender: int, receiver: int, bits: float) -> None:
        """Record directional payload load for communication diagnostics."""
        charged = float(bits) * self.vector_bit_multiplier
        self.per_agent_send_bits[sender] = self.per_agent_send_bits.get(sender, 0.0) + charged
        self.per_agent_receive_bits[receiver] = self.per_agent_receive_bits.get(receiver, 0.0) + charged
        key = (int(sender), int(receiver))
        self.pair_bits[key] = self.pair_bits.get(key, 0.0) + float(bits)

    @property
    def total_bits(self) -> float:
        return self.vector_bits + self.scalar_bits

    def as_dict(self) -> dict[str, float]:
        send = {str(k): float(v) for k, v in sorted(self.per_agent_send_bits.items())}
        receive = {str(k): float(v) for k, v in sorted(self.per_agent_receive_bits.items())}
        return {
            "vector_bits": self.vector_bits,
            "scalar_bits": self.scalar_bits,
            "total_bits": self.total_bits,
            "messages": float(self.messages),
            "matvec_rounds": float(self.rounds),
            "vector_bit_multiplier": float(self.vector_bit_multiplier),
            "pair_bits": {f"{k[0]}>{k[1]}": float(v) for k, v in sorted(self.pair_bits.items())},
            "per_agent_send_bits": send,
            "per_agent_receive_bits": receive,
            "max_agent_send_bits": max(send.values(), default=0.0),
            "max_agent_receive_bits": max(receive.values(), default=0.0),
        }


@dataclass
class DistributedOperator:
    """The aggregated matrix-vector product, as the agents can actually form it."""

    matrices: list          # per-region F_y^(k); dense or scipy.sparse
    owner: NDArray[np.int_]
    touch: tuple[frozenset[int], ...]
    channels: frozenset[tuple[int, int]]      # symmetrized (a, b) pairs allowed
    compressor: Compressor
    ledger: CommunicationLedger = field(default_factory=CommunicationLedger)
    _index: dict[tuple[int, int], NDArray[np.int_]] = field(default_factory=dict,
                                                            repr=False)

    def __post_init__(self) -> None:
        n_regions = len(self.matrices)
        size = self.owner.size
        buckets: dict[tuple[int, int], list[int]] = {}
        for i in range(size):
            k = int(self.owner[i])
            for l in self.touch[i]:
                buckets.setdefault((k, int(l)), []).append(i)
        for key, values in buckets.items():
            self._index[key] = np.asarray(values, dtype=int)
        self.size = size
        self.n_regions = n_regions

    def _allowed(self, a: int, b: int) -> bool:
        return a == b or (a, b) in self.channels or (b, a) in self.channels

    def matvec(self, vector: FloatArray) -> FloatArray:
        """One aggregated product. Accepts a vector or an (n, m) block.

        A block carries m right-hand sides through the SAME exchange, so the
        round count is unchanged and only the payload grows. That matters here:
        with heterogeneous regional objectives every agent has its own adjoint,
        and solving them one at a time would multiply the number of rounds by K
        for no reason.
        """
        vector = np.asarray(vector, dtype=np.float64)
        block = vector.ndim == 2
        width = vector.shape[1] if block else 1
        result = np.zeros(vector.shape)

        # ---- scatter: build each agent's view of the direction
        views = [np.zeros(vector.shape) for _ in range(self.n_regions)]
        for (k, l), index in self._index.items():
            if not self._allowed(k, l):
                continue
            if k == l:
                views[l][index] = vector[index]
                continue
            payload = self.compressor(vector[index], channel=f"s{k}->{l}")
            views[l][index] = payload
            bits = self.compressor.bits(index.size * width)
            self.ledger.record_vector_bits(bits)
            self.ledger.record_agent_bits(k, l, bits)
            self.ledger.messages += 1

        # ---- local products
        products = [
            np.asarray(self.matrices[l] @ views[l]).reshape(vector.shape)
            for l in range(self.n_regions)
        ]

        # ---- gather: owners aggregate the contributions they are allowed to hear
        for (k, l), index in self._index.items():
            if not self._allowed(k, l):
                continue
            if k == l:
                result[index] += products[l][index]
                continue
            payload = self.compressor(products[l][index], channel=f"g{l}->{k}")
            result[index] += payload
            bits = self.compressor.bits(index.size * width)
            self.ledger.record_vector_bits(bits)
            self.ledger.record_agent_bits(l, k, bits)
            self.ledger.messages += 1

        self.ledger.rounds += 1
        return result

    def aggregated_rhs(self, rhs_blocks: list[FloatArray]) -> FloatArray:
        """One aggregation of the right-hand side, under the same channel rules."""
        sample = np.asarray(rhs_blocks[0])
        width = sample.shape[1] if sample.ndim == 2 else 1
        result = np.zeros(sample.shape)
        for (k, l), index in self._index.items():
            if not self._allowed(k, l):
                continue
            if k == l:
                result[index] += rhs_blocks[l][index]
                continue
            payload = self.compressor(rhs_blocks[l][index], channel=f"b{l}->{k}")
            result[index] += payload
            bits = self.compressor.bits(index.size * width)
            self.ledger.record_vector_bits(bits)
            self.ledger.record_agent_bits(l, k, bits)
            self.ledger.messages += 1
        return result


@dataclass
class AdjointSolution:
    adjoint: FloatArray
    iterations: int
    residual: float
    residual_history: tuple[float, ...]
    ledger: CommunicationLedger
    converged: bool
    residual_vector: FloatArray | None = None   # CG's recursive residual r_R


def distributed_cg(operator: DistributedOperator, rhs: FloatArray,
                   max_rounds: int, tolerance: float = 1e-12,
                   preconditioner=None,
                   initial: FloatArray | None = None) -> AdjointSolution:
    """Truncated CG. ``max_rounds`` is the communication budget in matvecs.

    With ``initial`` (warm start) the first round forms b - A x0 and counts
    against the same budget.
    """

    rhs = np.asarray(rhs, dtype=np.float64)
    rhs_norm = float(np.linalg.norm(rhs)) or 1.0
    if initial is not None and np.any(initial) and max_rounds > 0:
        x = np.array(initial, dtype=np.float64)
        residual = rhs - operator.matvec(x)
        max_rounds -= 1
    else:
        x = np.zeros_like(rhs)
        residual = rhs.copy()

    def precondition(v: FloatArray) -> FloatArray:
        if preconditioner is None:
            return v
        if callable(preconditioner):
            return preconditioner(v)          # e.g. the owned entropy blocks
        return v * preconditioner

    z = precondition(residual)
    direction = z.copy()
    rz = float(residual @ z)
    history = [float(np.linalg.norm(residual)) / rhs_norm]

    iterations = 0
    for _ in range(max_rounds):
        if history[-1] <= tolerance:
            break
        q = operator.matvec(direction)
        operator.ledger.scalar_bits += 64.0 * operator.n_regions  # p^T q partials
        denominator = float(direction @ q)
        if denominator <= 0.0:
            break                     # lost definiteness: compression broke the operator
        alpha = rz / denominator
        x += alpha * direction
        residual -= alpha * q
        iterations += 1
        history.append(float(np.linalg.norm(residual)) / rhs_norm)
        z = precondition(residual)
        rz_next = float(residual @ z)
        operator.ledger.scalar_bits += 64.0 * operator.n_regions  # r^T z partials
        direction = z + (rz_next / rz) * direction
        rz = rz_next

    return AdjointSolution(
        adjoint=x,
        iterations=iterations,
        residual=history[-1],
        residual_history=tuple(history),
        ledger=operator.ledger,
        converged=history[-1] <= tolerance,
        residual_vector=residual.copy(),
    )


def distributed_block_cg(operator: DistributedOperator, rhs: FloatArray,
                         max_rounds: int, tolerance: float = 1e-12,
                         preconditioner=None,
                         initial: FloatArray | None = None
                         ) -> AdjointSolution:
    """Truncated CG on several right-hand sides sharing every exchange.

    Each column runs its own scalar recursion; the communication is shared, so
    ``max_rounds`` remains the number of ROUNDS, not the number of solves. This
    is what keeps heterogeneous objectives affordable: K adjoints cost K times
    the payload but the same number of rounds as one.
    """
    rhs = np.atleast_2d(np.asarray(rhs, dtype=np.float64))
    if rhs.shape[0] == 1 and rhs.size != operator.size:
        rhs = rhs.T
    norms = np.linalg.norm(rhs, axis=0)
    norms[norms == 0.0] = 1.0
    if initial is not None and np.any(initial) and max_rounds > 0:
        x = np.array(initial, dtype=np.float64).reshape(rhs.shape)
        residual = rhs - operator.matvec(x)          # warm start: one counted round
        max_rounds -= 1
    else:
        x = np.zeros_like(rhs)
        residual = rhs.copy()

    def precondition(v):
        if preconditioner is None:
            return v
        if callable(preconditioner):
            return preconditioner(v)                  # owned entropy blocks
        return v * preconditioner[:, None]

    z = precondition(residual)
    direction = z.copy()
    rz = np.einsum("ij,ij->j", residual, z)
    history = [float(np.max(np.linalg.norm(residual, axis=0) / norms))]

    iterations = 0
    for _ in range(max_rounds):
        if history[-1] <= tolerance:
            break
        # Columns that are already solved -- including right-hand sides that are
        # identically zero, e.g. a jurisdiction with no resident demand -- carry
        # a zero search direction. They must be frozen, not allowed to trip the
        # definiteness guard: a zero denominator there is not a loss of
        # definiteness, and breaking on it silently stopped EVERY column.
        active = np.linalg.norm(residual, axis=0) / norms > tolerance
        if not np.any(active):
            break
        q = operator.matvec(direction)
        operator.ledger.scalar_bits += 64.0 * operator.n_regions * int(active.sum())
        denominator = np.einsum("ij,ij->j", direction, q)
        if np.any(denominator[active] <= 0.0):
            break                     # genuine loss of definiteness on a live column
        alpha = np.where(active, rz / np.where(active, denominator, 1.0), 0.0)
        x += alpha * direction
        residual -= alpha * q
        iterations += 1
        history.append(float(np.max(np.linalg.norm(residual, axis=0) / norms)))
        z = precondition(residual)
        rz_next = np.einsum("ij,ij->j", residual, z)
        operator.ledger.scalar_bits += 64.0 * operator.n_regions * int(active.sum())
        safe = np.where(np.abs(rz) > 0.0, rz, 1.0)
        beta = np.where(active, rz_next / safe, 0.0)
        direction = np.where(active, z + beta * direction, 0.0)
        rz = rz_next

    return AdjointSolution(
        adjoint=x, iterations=iterations, residual=history[-1],
        residual_history=tuple(history), ledger=operator.ledger,
        converged=history[-1] <= tolerance,
    )


def local_jacobi_preconditioner(matrices: list,
                                owner: NDArray[np.int_]) -> FloatArray:
    """Each agent scales by its OWN diagonal only. Requires no communication."""
    size = owner.size
    diagonal = np.zeros(size)
    for region, matrix in enumerate(matrices):
        owned = np.where(owner == region)[0]
        full = matrix.diagonal() if hasattr(matrix, "diagonal") else np.diag(matrix)
        diagonal[owned] = np.asarray(full).ravel()[owned]
    diagonal[diagonal <= 0.0] = 1.0
    return 1.0 / diagonal
