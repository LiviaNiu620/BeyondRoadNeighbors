"""Numerical checks of the paper's structural results on a small instance."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from conftest import n_norm


def test_regional_decomposition(instance):
    """Proposition 1 and Eq. (2): A = sum_k A_k = C + N with N owner-held and positive definite."""
    S, A, N = instance["systems"], instance["A"], instance["N"]
    C = sum(s.congestion for s in S)
    assert abs(A - (C + N)).max() < 1e-9 * abs(A).max()
    assert np.all(np.linalg.eigvalsh(N.toarray()) > 0)


def test_entropy_preconditioner_inverse(instance):
    """The owner-local Sherman-Morrison apply is the exact inverse of N."""
    N, P = instance["N"], instance["P"]
    x = np.random.default_rng(0).standard_normal(N.shape[0])
    assert np.linalg.norm(N @ P.apply(x) - x) < 1e-9 * np.linalg.norm(x)


def test_spectral_enclosure(instance):
    """Theorem 4: spec(N^{-1}A) lies in [1, 1 + gamma] with the computable path bound gamma."""
    from brn.spectral import sharpened_gamma
    A, N, dec = instance["A"].toarray(), instance["N"].toarray(), instance["dec"]
    w = np.linalg.eigvals(np.linalg.solve(N, A)).real
    gamma = sharpened_gamma(instance["problem"], dec.link_region, instance["hdv"], instance["av"], 4).gamma_path
    assert w.min() > 1 - 1e-8
    assert w.max() <= 1 + gamma + 1e-8


def _masked(instance, pairs):
    S, own, b = instance["systems"], instance["dec"].coordinate_owner, instance["b"]
    At, bt = None, np.zeros_like(b)
    for k, s in enumerate(S):
        m = np.array([(own[i] == k) or ((int(own[i]), k) in pairs) for i in range(own.size)], float)
        R = sparse.diags(m)
        t = R @ s.matrix @ R
        At = t if At is None else At + t
        bt += m * s.rhs
    return At.tocsr(), bt


def test_omission_theorem_and_coverage(instance):
    """Theorem 2 / Corollary 3: a mask's fixed-point error is pinned by the omitted flux;
    covering G_dep gives zero error."""
    from scipy.sparse.linalg import eigsh, LinearOperator
    A, b, lam, N, P, dec = (instance[k] for k in ("A", "b", "lam", "N", "P", "dec"))
    road = set()
    for (i, j) in dec.road_adjacency:
        road |= {(i, j), (j, i)}
    At, bt = _masked(instance, road)
    lt = spsolve(At.tocsc(), bt)
    phi = (A - At) @ lam - (b - bt)
    flux = float(np.sqrt(phi @ P.apply(phi)))
    Minv = LinearOperator(N.shape, matvec=P.apply, dtype=float)
    mu = eigsh(At, k=1, M=N, Minv=Minv, which="LA", return_eigenvectors=False, tol=1e-10)[0]
    err = n_norm(N, lt - lam)
    assert flux / mu <= err * (1 + 1e-9) and err <= flux * (1 + 1e-9)
    full = {(s_, r_) for (r_, s_) in dec.coupling}
    Af, bf = _masked(instance, full)
    assert np.linalg.norm(spsolve(Af.tocsc(), bf) - lam) < 1e-8 * np.linalg.norm(lam)


def test_a_posteriori_certificate(instance):
    """Lemma 7: ||x - lambda||_N <= ||b - A x||_{N^{-1}} for any x."""
    A, b, lam, N, P = (instance[k] for k in ("A", "b", "lam", "N", "P"))
    rng = np.random.default_rng(1)
    for scale in (1e-3, 1e-1, 1.0):
        x = lam + scale * np.linalg.norm(lam) * rng.standard_normal(lam.size) / np.sqrt(lam.size)
        r = b - A @ x
        assert n_norm(N, x - lam) <= float(np.sqrt(r @ P.apply(r))) * (1 + 1e-12)


def test_epda_cg_with_full_coverage(instance):
    """Distributed preconditioned CG over G_dep converges to the centralized adjoint."""
    from brn.distributed import CommunicationLedger, DistributedOperator, distributed_cg
    from brn.compression import build_compressor
    from brn.partition import communication_graph
    S, dec, b, lam, P = (instance[k] for k in ("systems", "dec", "b", "lam", "P"))
    op = DistributedOperator(matrices=[s.matrix for s in S], owner=dec.coordinate_owner,
                             touch=dec.coordinate_touch, channels=communication_graph(dec, "coupling"),
                             compressor=build_compressor("float64", seed=0), ledger=CommunicationLedger())
    sol = distributed_cg(op, b, 64, tolerance=1e-12, preconditioner=P.apply)
    assert np.linalg.norm(sol.adjoint - lam) < 1e-8 * np.linalg.norm(lam)
