# Run records

These files are the evidence behind the paper's tables, figures and text. `paper/make_numbers.py`
reads them directly; the experiment scripts never write here (new runs go to `outputs/`). Each
`.jsonl` file holds one JSON record per protocol.

| Directory | Content |
|---|---|
| `revision/` | Final runs on one ledger: masked protocols use the entropy preconditioner, warm starts and a relative residual of 1e-6 and are charged once per road hop; CG protocols marked `+res` form the AV gradient as N λ_R + r_R. Table 1 (`t1_*`, `h3_*`), Table 3 (`res_*`, `nb_*`), Figures 1(b), 2 and 4 (`figs/`), the channel-failure loop, the gradient certificate (`gradient_corollary_*`). |
| `extended/` | κ-hop masks under the omission theorem (`omission_khop_*`), Armijo line search (`r_*_armijo_*`), second-order Richardson at certified rounds (`r_*_richP_*`), spectral bounds at high load, the residual-gap certificate (`runtime_bound_*`). |
| `submission/` | Runs of the submitted version that the final paper still uses: Table 2 (`risk_omission_*`), Figure 3 (`risk_locality_Anaheim`, `risk_gamma_share`), Figure 5 (`risk_failures_Anaheim_K16`), state broadcast, coordinator and drivers-held-fixed rows of Tables 1 and 3, fire-zone and weight sensitivity, perturbed starts, per-authority alignment, spectral bounds at θ = 0.08 and 0.32, and the channel-failure draws (`failures_Anaheim_K16`). |
| `tstt/` | The same decision loop under total system travel time (ρ = 0), quoted in Appendix D.5. |

## Regimes and file names

`base` = nominal (θ = 0.08), `theta032` = drivers four times more cost-sensitive (θ = 0.32),
`high` = θ = 0.32 with doubled demand. All runs use the exposure-weighted objective with ρ = 2,
ℓ = 2 and the fire zone at the largest-demand region of the 16-way partition unless the name
says otherwise (`fire<r>` = fire zone at region r, `rho<x>` = ρ = x, `seed<s>` = perturbed start
s, `K<k>` = k authorities, `SF8` = SiouxFalls with 8 authorities).

## Decision-loop records (`decision_loop.py`)

| Field | Meaning |
|---|---|
| `config` | protocol label (see `experiments/README.md`) |
| `initial_tstt`, `tstt` | objective J at the start and after 30 attempts (the name is historical: under the exposure objective these fields hold the weighted J) |
| `tstt_improvement` | relative change of J; retention = this value divided by the reference's |
| `total_bits` | payload over all attempts on direct channels, including 64-bit scales and scalar reductions |
| `road_hop_charged_bits` | the same payload charged once per road hop (masked protocols and relays) |
| `accepted_steps`, `attempt_steps` | steps that lowered J, and attempts made |
| `per_agent_send_bits`, `per_agent_receive_bits` | communication load per authority |
| `tstt_trace`, `bits_trace` | J and cumulative payload after every attempt |
| `gradient_form` | `residual` (N λ_R + r_R, no final exchange) or `congestion` (b − C λ_R after a final product exchange) |

## Single-state records

`omission_theorem.py` writes, per partition and mask: the protocol's own residual
(`internal_residual`), the true adjoint error in the N-norm (`err_N`) and Euclidean norm
(`err_2`), the omitted flux, λ_max(N⁻¹Ã), and the lower and upper bounds of Theorem 2, all
relative to ‖λ‖_N. The other probes document their fields in their module docstrings.
