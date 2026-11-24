"""Shared small instance: SiouxFalls, K=4 regions, initial controls, total-travel-time objective."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve


@pytest.fixture(scope="session")
def instance():
    from brn.experiments.common import load, follower, regions
    from brn import assemble_local_systems, centralized_system
    from brn.spectral import entropy_preconditioner

    problem, coords, av, logits = load("SiouxFalls")
    hdv = follower(problem, coords, av, logits)
    dec = regions(problem, coords, 4)
    systems = assemble_local_systems(problem, coords, dec, hdv, av)
    A, b = centralized_system(systems)
    A = A.tocsr()
    lam = spsolve(A.tocsc(), b)
    P = entropy_preconditioner(problem, hdv)
    return dict(problem=problem, coords=coords, av=av, hdv=hdv, dec=dec, systems=systems,
                A=A, b=b, lam=lam, P=P, N=P.matrix().tocsr())


def n_norm(N, x):
    return float(np.sqrt(x @ (N @ x)))
