# Experiments

Run every script through `run.py`, which fixes the objective and the load before the script
builds its instance:

```bash
python experiments/run.py [--rho 2] [--demand-scale 1] <script.py> [script options]
```

`--rho` sets the exposure weights $w_a = 1+\rho e^{-\mathrm{hop}_a/2}$ (ρ = 0 gives total
system travel time); `--demand-scale 2` doubles every trip-table entry. `RISK_FIRE=<region>`
moves the fire zone and `RISK_NETWORK` / `RISK_K` choose the partition that defines it. Outputs
go where `--out` (or `OUT=` for the scripts that use it) points, by default `outputs/`.
The shell scripts in `jobs/` rerun whole tables; see the main README.

## Decision loops (`decisions/`)

| Script | What it runs | Paper |
|---|---|---|
| `decision_loop.py` | 30 projected AV and guidance steps on the team gradient under the protocols in `--only` | Tables 1, 3; App. D.1, D.3–D.5 |
| `state_sharing_loop.py` | the same loop with state broadcast (`allgather-*`) or a coordinator (`coord-*`) | Tables 1, 3 |
| `drivers_fixed_loop.py` | complete coverage but drivers held fixed (no equilibrium response) | Table 1 |
| `channel_failure_loop.py` | relay versus neighbor-only delivery on drawn channel failures | Sec. 6.5 |

Protocol labels for `decision_loop.py --only` (comma-separated):

| Label | Protocol |
|---|---|
| `exact`, `exactN` | full-precision adjoint over $G_{\rm dep}$ (Jacobi / entropy preconditioner); `exactN` is the reference of Tables 1 and 3 |
| `N:q8+r2+warm+res` | EPDA-CG: 8-bit messages, 2 warm-started rounds, AV gradient $N\lambda_R+r_R$ |
| `N:q8+r2+warm` | the same with the final product exchange for the AV gradient (submitted algorithm) |
| `relay:N:q8+r2+warm+res` | EPDA products relayed hop by hop over road channels |
| `roadN@1e-6+warm+res` | neighbor-only mask, entropy-preconditioned CG to a relative residual of 1e-6 |
| `road2N@1e-6+warm+res`, `road3N@1e-6+warm+res` | two- and three-hop masks, same solver |
| `road`, `road2`, `road3` | masks run for 64 full-precision rounds (submission ledger) |
| `RichP:q8+r5+warm` | second-order Richardson with the path bound γ, 5 rounds (Proposition 6(ii)) |

`--line-search armijo` replaces the halving rule by per-attempt backtracking;
`--start perturbed --seed s` draws a perturbed start; `--theta` sets driver cost sensitivity.

## Single-state probes (`single_state/`)

| Script | Measures | Paper |
|---|---|---|
| `silent_failure_trace.py` | per-round own residual vs true error, neighbor-only vs complete | Fig. 1(b) |
| `omission_theorem.py` | Theorem 2: error, omitted flux and bounds of each mask | Table 2, App. D.3 |
| `coverage_vs_precision.py` | coverage × precision frontier; direct vs response omission; flux ranking from EPDA's own adjoint | Fig. 2, Sec. 6.3 |
| `locality.py` | influence by road distance for each ordered pair of authorities | Fig. 3(a) |
| `congestion_correction.py` | relative congestion-induced adjoint correction vs γ* | Fig. 3(b), Prop. 5 |
| `response_share.py` | direct and response parts of the omitted flux across loads | Sec. 6.3 |
| `fire_zone_flux.py`, `fire_zone_sensitivity.py` | fire-zone share of the dependency flux; five fire zones × three weights | Sec. 6.3, 6.5 |
| `per_authority_alignment.py` | cosine of each authority's gradient block with the exact one | Sec. 6.2 |
| `protocol_frontier.py` | gradient error vs payload for coordinator, broadcast and EPDA | Fig. 4 |
| `channel_failures.py` | adjoint error under random road-channel failures, relay vs neighbor-only | Fig. 5 |
| `gradient_direction.py`, `flux_ranked_pruning.py`, `exposure_single_state.py` | supporting checks of gradient direction, flux-ranked pruning and the exposure objective | Sec. 6.2–6.3 |

## Certificates (`certificates/`)

| Script | Measures | Paper |
|---|---|---|
| `spectral_bounds.py` | γ*, the path bound γ and its regional relaxation; fixed-round solver accuracy | Table `tab:cert` |
| `gradient_certificate.py` | Lemma 7 (a posteriori) and Corollary 8 (gradient error) on 8-bit CG | Table `tab:apost` |
| `residual_gap_certificate.py` | residual-gap certificate for quantized CG | App. D.2 |
| `quantization_and_route_length.py` | size of the quantization term of Proposition 6(ii); regions crossed per route | App. D.5 |
| `spectrum.py` | spectrum of $N^{-1}A$ and solver accuracy by round | Sec. 5 |

## Provenance

These scripts are the research scripts that produced `results/`, with module paths and file
names cleaned up; numerical code is unchanged. Earlier names: `decision_loop.py` ←
`run_v2.py`, `state_sharing_loop.py` ← `run_state_sharing_outer.py`, `drivers_fixed_loop.py` ←
`run_unaware_outer.py`, `channel_failure_loop.py` ← `run_failure_outer.py`, single-state and
certificate scripts ← `probe_*.py`, `run.py` ← `run_risk.py`, `brn.experiments.exposure` ←
`risk.py`, `brn.experiments.demand` ← `scaled.py`. The `.log` files in `results/` show the
original commands.
