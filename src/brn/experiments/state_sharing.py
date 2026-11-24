"""State-sharing baselines: coordinator upload and peer state broadcast.

A coordinator, or every authority (all-gather), rebuilds the full adjoint system
from the raw local state and solves it exactly. The state that determines
A, b and C is exactly what EPDA's agents already observe:

  * link flows v on the links of region k   (sent by region k);
  * HDV path flows f of the groups owned by k (sent by owner k).

Path library, incidence, BPR parameters, theta and ownership are pre-shared, as
in EPDA. Messages use the same compressor family and the same bit accounting
(payload + 64-bit scale header for uniform quantizers).

Protocols
  coord-<fmt>     : K uploads of (v_k, f_k); the coordinator returns lambda and
                    grad_a on each owner's coordinates (same format).
  allgather-<fmt> : every authority broadcasts (v_k, f_k) to the K-1 others and
                    solves locally. No download.

Formats: float64, float32, float16, uniformB (EPDA's quantizer; a decoded
non-positive path flow is floored at half a step), logB (uniform in log f over
[min, max] with two 64-bit headers; positive data only, v uses log(1+v)).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from brn.compression import build_compressor


# ---------------------------------------------------------------- formats
class Format:
    def __init__(self, spec: str):
        self.spec = spec
        if spec.startswith("uniform"):
            self.q = build_compressor(spec)
        else:
            self.q = None

    def bits(self, size: int) -> float:
        s = self.spec
        if size == 0:
            return 0.0
        if s == "float64":
            return 64.0 * size
        if s == "float32":
            return 32.0 * size
        if s == "float16":
            return 16.0 * size
        if s.startswith("uniform"):
            return self.q.bits(size)
        if s.startswith("log"):
            return int(s[3:]) * size + 128.0
        raise ValueError(s)

    def __call__(self, x: np.ndarray, positive: bool = False, log1p: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        s = self.spec
        if x.size == 0:
            return x.copy()
        if s == "float64":
            return x.copy()
        if s == "float32":
            return x.astype(np.float32).astype(np.float64)
        if s == "float16":
            return x.astype(np.float16).astype(np.float64)
        if s.startswith("uniform"):
            y = self.q(x, channel="state")
            if positive:
                scale = float(np.max(np.abs(x)))
                levels = 2 ** (self.q.n_bits - 1) - 1
                y = np.maximum(y, 0.5 * scale / levels)
            return y
        if s.startswith("log"):
            nb = int(s[3:])
            z = np.log1p(x) if log1p else np.log(x)
            lo, hi = float(z.min()), float(z.max())
            if hi == lo:
                zq = z
            else:
                levels = 2 ** nb - 1
                zq = lo + np.round((z - lo) / (hi - lo) * levels) * (hi - lo) / levels
            return np.expm1(zq) if log1p else np.exp(zq)
        raise ValueError(s)


# ---------------------------------------------------------------- system from state
class StateSystem:
    """Builds (A, b, C) from link flows v and HDV path flows f."""

    def __init__(self, problem, coordinates):
        from brn.sue.operators import sparse_incidence_matrix
        from brn.sue.sparse_sensitivity import sparse_reduced_basis
        self.problem = problem
        self.coordinates = coordinates
        self.D = sparse_incidence_matrix(problem).tocsr()
        self.E = sparse_reduced_basis(coordinates).tocsc()
        gid = {g.group_id: g for g in problem.groups}
        self.theta = np.asarray([gid[p.group_id].cost_sensitivity for p in problem.paths])
        self.DE = (self.D @ self.E).tocsr()

    def build(self, v, f):
        from brn.sue.operators import bpr_link_costs, bpr_link_derivatives
        tp = bpr_link_derivatives(self.problem, v)
        m = bpr_link_costs(self.problem, v) + v * tp
        C = (self.DE.T @ sparse.diags(tp) @ self.DE).tocsr()
        N = (self.E.T @ sparse.diags(1.0 / (self.theta * f)) @ self.E).tocsr()
        b = np.asarray(self.DE.T @ m).ravel()
        return (C + N).tocsr(), b, C

    def gradient(self, v, f, logits):
        from brn.outer import _logit_gradient
        A, b, C = self.build(v, f)
        lam = spsolve(A.tocsc(), b)
        return lam, b - C @ lam, _logit_gradient(self.coordinates, self.problem, logits, lam)


# ---------------------------------------------------------------- protocols
def state_messages(decomposition):
    """Index sets of the two messages each authority k contributes."""
    link_region = np.asarray(decomposition.link_region)
    path_owner = np.asarray(decomposition.path_owner)
    K = decomposition.n_regions
    return [(np.where(link_region == k)[0], np.where(path_owner == k)[0]) for k in range(K)]


def protocol_bits(kind: str, fmt: Format, decomposition) -> dict:
    K = decomposition.n_regions
    msgs = state_messages(decomposition)
    upload = sum(fmt.bits(l.size) + fmt.bits(p.size) for l, p in msgs)
    owner = np.asarray(decomposition.coordinate_owner)
    per_agent_state = [fmt.bits(l.size) + fmt.bits(p.size) for l, p in msgs]
    dfmt = Format("uniform" + fmt.spec[3:]) if fmt.spec.startswith("log") else fmt
    if kind == "coord":
        download = sum(2 * dfmt.bits(int(np.sum(owner == k))) for k in range(K))
        total = upload + download
        max_send = max(per_agent_state)
        max_recv = max(2 * dfmt.bits(int(np.sum(owner == k))) for k in range(K))
        coordinator_load = upload + download
    elif kind == "allgather":
        total = (K - 1) * upload
        max_send = (K - 1) * max(per_agent_state)
        max_recv = max(upload - s for s in per_agent_state)
        coordinator_load = 0.0
    else:
        raise ValueError(kind)
    return dict(total_bits=float(total), max_agent_send_bits=float(max_send),
                max_agent_receive_bits=float(max_recv),
                coordinator_bits=float(coordinator_load))


def shared_gradient(kind: str, fmt: Format, system: StateSystem, decomposition,
                    v, f, logits):
    """Gradient obtained by the protocol (quantized uploads, and downloads for coord)."""
    msgs = state_messages(decomposition)
    vq = np.zeros_like(v)
    fq = np.zeros_like(f)
    for links, paths in msgs:
        if links.size:
            vq[links] = fmt(v[links], log1p=True)
        if paths.size:
            fq[paths] = fmt(f[paths], positive=True)
    lam, ga, _ = system.gradient(vq, fq, logits)
    if kind == "coord" and fmt.spec != "float64":
        owner = np.asarray(decomposition.coordinate_owner)
        lam_d = np.zeros_like(lam)
        ga_d = np.zeros_like(ga)
        dfmt = Format("uniform" + fmt.spec[3:]) if fmt.spec.startswith("log") else fmt
        for k in range(decomposition.n_regions):
            idx = np.where(owner == k)[0]
            if idx.size:
                lam_d[idx] = dfmt(lam[idx])
                ga_d[idx] = dfmt(ga[idx])
        lam, ga = lam_d, ga_d
    from brn.outer import _logit_gradient
    gu = _logit_gradient(system.coordinates, system.problem, logits, lam)
    return lam, ga, gu
