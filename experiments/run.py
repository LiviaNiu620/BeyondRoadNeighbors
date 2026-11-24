"""Run an experiment script under the paper's objective and load setting.

    python experiments/run.py [--rho RHO] [--demand-scale S] <script.py> [script args...]

--rho           exposure weight rho in w_a = 1 + rho * exp(-hop_a / 2); 0 gives TSTT (default 2)
--demand-scale  multiply every trip-table entry (2 = the paper's high load; default 1)

Environment: RISK_FIRE selects the fire-zone region (default: largest originating demand),
RISK_NETWORK / RISK_K pick the network and partition that define the fire zone
(SiouxFalls runs use RISK_NETWORK=SiouxFalls RISK_K=8).
"""
import runpy
import sys
from pathlib import Path

args = sys.argv[1:]
rho = 2.0
scale = 1.0
while args and args[0] in ("--rho", "--demand-scale"):
    if args[0] == "--rho":
        rho = float(args[1])
    else:
        scale = float(args[1])
    args = args[2:]
if not args:
    sys.exit(__doc__)

from brn.experiments import common  # noqa: E402,F401
from brn.experiments import exposure as risk, demand as scaled  # noqa: E402

risk.apply(rho=rho)
scaled.apply(scale)
script = Path(args[0]).resolve()
sys.path.insert(0, str(script.parent))
sys.argv = [str(script)] + args[1:]
runpy.run_path(str(script), run_name="__main__")
