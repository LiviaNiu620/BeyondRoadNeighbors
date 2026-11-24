"""Unit checks of the quantizer, the simplex projection and the paper-number generator."""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_uniform_quantizer_error_and_bits():
    from brn.compression import UniformQuantizer
    rng = np.random.default_rng(0)
    for nu in (4, 6, 8, 16):
        q = UniformQuantizer(n_bits=nu)
        x = rng.standard_normal(1000)
        L = 2 ** (nu - 1) - 1
        assert np.max(np.abs(q(x, "c") - x)) <= np.max(np.abs(x)) / (2 * L) * (1 + 1e-12)
        assert q.bits(1000) == nu * 1000 + 64        # payload plus a 64-bit scale


def test_weighted_simplex_projection_feasible():
    from brn.sue.projection import weighted_lower_simplex_projection
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 6))
        point = rng.normal(0, 50, n)
        lower = rng.uniform(0, 1, n)
        total = float(lower.sum() + rng.uniform(0.1, 10))
        x = weighted_lower_simplex_projection(point, total, lower, rng.uniform(0.01, 100, n))
        assert np.all(x >= lower - 1e-9)
        assert abs(x.sum() - total) < 1e-8 * max(1.0, total)


def test_paper_numbers_regenerate(tmp_path):
    """paper/make_numbers.py reproduces the paper's numbers.tex and appendix_results.tex."""
    subprocess.run([sys.executable, str(ROOT / "paper/make_numbers.py"), str(ROOT / "results"), str(tmp_path)],
                   check=True, capture_output=True)
    for name in ("numbers.tex", "appendix_results.tex"):
        assert (tmp_path / name).read_text() == (ROOT / "paper/expected" / name).read_text()
