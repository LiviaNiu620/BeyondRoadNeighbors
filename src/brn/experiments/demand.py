"""Demand scaling hook: `import scaled; scaled.apply(2.0)` before building problems."""
import brn.sue.io.tntp as tn
_orig = tn.read_tntp_demands
def apply(scale):
    if scale == 1.0:
        tn.read_tntp_demands = _orig; return
    tn.read_tntp_demands = lambda *a, **k: tuple(tn.ODDemandRecord(r.origin, r.destination, r.demand * scale) for r in _orig(*a, **k))
