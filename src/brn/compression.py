"""Compression operators for the messages exchanged during the adjoint solve.

Every operator reports the number of bits the message would occupy, so a run can
report an optimality gap against a communication budget rather than against an
iteration count. Float64 is the uncompressed reference at 64 bits per entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Compressor:
    """Base class: transmit exactly, count 64 bits per entry."""

    name: str = "float64"

    def bits(self, size: int) -> float:
        return 64.0 * size

    def __call__(self, vector: FloatArray, channel: str) -> FloatArray:
        return np.asarray(vector, dtype=np.float64)

    def reset(self) -> None:
        return None


@dataclass
class UniformQuantizer(Compressor):
    """Deterministic midtread quantization on [-s, s], s = max|x| sent as a header."""

    n_bits: int = 8

    def __post_init__(self) -> None:
        if self.n_bits < 1:
            raise ValueError("n_bits must be at least 1")
        self.name = f"uniform{self.n_bits}"

    def bits(self, size: int) -> float:
        return self.n_bits * size + 64.0  # payload plus the scale header

    def __call__(self, vector: FloatArray, channel: str) -> FloatArray:
        x = np.asarray(vector, dtype=np.float64)
        scale = float(np.max(np.abs(x))) if x.size else 0.0
        if scale == 0.0:
            return np.zeros_like(x)
        levels = 2 ** (self.n_bits - 1) - 1
        if levels < 1:                      # 1 bit -> sign times scale
            return scale * np.sign(x)
        step = scale / levels
        return np.clip(np.round(x / step), -levels, levels) * step


@dataclass
class StochasticQuantizer(UniformQuantizer):
    """Unbiased randomized rounding (QSGD-style). Unbiasedness helps error feedback."""

    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.name = f"stochastic{self.n_bits}"
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, vector: FloatArray, channel: str) -> FloatArray:
        x = np.asarray(vector, dtype=np.float64)
        scale = float(np.max(np.abs(x))) if x.size else 0.0
        if scale == 0.0:
            return np.zeros_like(x)
        levels = max(2 ** (self.n_bits - 1) - 1, 1)
        step = scale / levels
        scaled = x / step
        floor = np.floor(scaled)
        probability = scaled - floor
        rounded = floor + (self._rng.random(x.shape) < probability)
        return np.clip(rounded, -levels, levels) * step


@dataclass
class TopK(Compressor):
    """Keep the largest-magnitude fraction; pay value bits plus index bits."""

    ratio: float = 0.1
    value_bits: int = 32

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError("ratio must lie in (0, 1]")
        self.name = f"top{self.ratio:g}"

    def _kept(self, size: int) -> int:
        return max(1, int(math.ceil(self.ratio * size))) if size else 0

    def bits(self, size: int) -> float:
        kept = self._kept(size)
        index_bits = math.ceil(math.log2(size)) if size > 1 else 0
        return kept * (self.value_bits + index_bits)

    def __call__(self, vector: FloatArray, channel: str) -> FloatArray:
        x = np.asarray(vector, dtype=np.float64)
        if x.size == 0:
            return x
        kept = self._kept(x.size)
        out = np.zeros_like(x)
        index = np.argpartition(np.abs(x), x.size - kept)[x.size - kept:]
        out[index] = x[index]
        return out


@dataclass
class ErrorFeedback(Compressor):
    """Wrap a compressor, carrying each channel's residual into the next message."""

    inner: Compressor = field(default_factory=Compressor)
    _memory: dict[str, FloatArray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.name = f"ef({self.inner.name})"

    def bits(self, size: int) -> float:
        return self.inner.bits(size)

    def __call__(self, vector: FloatArray, channel: str) -> FloatArray:
        x = np.asarray(vector, dtype=np.float64)
        memory = self._memory.get(channel)
        if memory is None or memory.shape != x.shape:
            memory = np.zeros_like(x)
        corrected = x + memory
        sent = self.inner(corrected, channel)
        self._memory[channel] = corrected - sent
        return sent

    def reset(self) -> None:
        self._memory.clear()
        self.inner.reset()


def build_compressor(spec: str, seed: int = 0) -> Compressor:
    """``float64`` | ``uniformB`` | ``stochasticB`` | ``topR`` | ``ef:<spec>``."""
    if spec.startswith("ef:"):
        return ErrorFeedback(inner=build_compressor(spec[3:], seed))
    if spec == "float64":
        return Compressor()
    if spec.startswith("uniform"):
        return UniformQuantizer(n_bits=int(spec[len("uniform"):]))
    if spec.startswith("stochastic"):
        return StochasticQuantizer(n_bits=int(spec[len("stochastic"):]), seed=seed)
    if spec.startswith("top"):
        return TopK(ratio=float(spec[len("top"):]))
    raise ValueError(f"unknown compressor spec {spec!r}")
