"""DC-SIIG: the decentralized outer loop.

Each agent owns the control block of the demand groups originating in its region
-- the reduced AV path flows and the reference-normalized recommendation logits --
and updates only that block. Per outer iteration:

  1. the follower equilibrium is realized by the physical network (observed, not
     communicated);
  2. each agent assembles F_y^(k), F_a^(k) and J_y^(k) from its own links;
  3. the agents run compressed truncated CG for the adjoint  (COMMUNICATION);
  4. each agent aggregates F_a lambda on the coordinates it owns  (COMMUNICATION);
  5. each agent forms its own gradient block and takes a feasible projected step.

The recommendation gradient  -F_u^T lambda  needs lambda only on coordinates of
the agent's own groups, so it is entirely local once lambda is in hand. The AV
gradient  J_y - F_a^T lambda  needs one extra aggregation, which reuses the same
channels and the same compressor as the CG matvec.

Everything is counted by the same ledger, so reported finite-budget performance
comes with the total number of bits used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .assembly import assemble_local_systems
from .compression import build_compressor
from .distributed import (
    CommunicationLedger,
    DistributedOperator,
    distributed_cg,
    local_jacobi_preconditioner,
)
from .partition import communication_graph
from .objective_contract import result_metadata

FloatArray = NDArray[np.float64]


@dataclass
class DCSIIGOptions:
    topology: str = "coupling"
    compressor: str = "float64"
    cg_rounds: int = 64
    cg_tolerance: float = 1e-13
    outer_iterations: int = 40
    av_step: float = 1.0
    logit_step: float = 0.05
    logit_bound: float = 2.0
    minimum_av_flow: float = 1e-6
    follower_tolerance: float = 1e-10
    follower_max_iterations: int = 200_000
    backtrack: float = 0.5          # shrink both steps when TSTT increases
    step_decay_tau: float = 50.0    # eta_t = eta_0 / (1 + t / tau); 0 disables
    seed: int = 0
    communication_budget_bits: float | None = None
    relay_multiplier: float = 1.0
    full_diagnostics: bool = False
    # ---- v2 (defaults reproduce v1 exactly)
    solver: str = "cg_jacobi"        # cg_jacobi | cg_entropy | richardson
    warm_start: bool = False
    gamma_bound: str = "sharp_regional"   # sharp_regional | sharp_path (Richardson interval)
    count_rhs: bool = False          # game: charge delivery of regional right-hand sides
    # ---- residual-form gradient, line search, kappa-hop masks (defaults reproduce the submitted runs)
    gradient_form: str = "congestion"   # congestion: b - C lam (one more scatter-gather);
                                        # residual: N lam + r_R from CG, no extra exchange
    line_search: str = "halve"          # halve: submitted rule (reject -> halve, never re-expand)
                                        # armijo: each attempt restarts from the initial step and
                                        # backtracks within the attempt until J decreases
    max_backtracks: int = 30


@dataclass
class DCSIIGTrace:
    tstt: list[float] = field(default_factory=list)
    stationarity: list[float] = field(default_factory=list)
    adjoint_residual: list[float] = field(default_factory=list)
    bits: list[float] = field(default_factory=list)
    av_step: list[float] = field(default_factory=list)
    movement: list[float] = field(default_factory=list)
    full_online_adjoint_residual: list[float] = field(default_factory=list)
    full_online_av_mapping: list[float] = field(default_factory=list)
    full_online_logit_gradient_inf: list[float] = field(default_factory=list)
    gamma_hat: list[float] = field(default_factory=list)
    follower_solves: int = 0
    attempt_steps: list[float] = field(default_factory=list)   # av step tried at each attempt
    stalled: bool = False      # armijo: a full backtrack from one state found no decrease


@dataclass
class DCSIIGResult:
    av_flow: FloatArray
    logits: FloatArray
    tstt: float
    initial_tstt: float
    best_tstt: float
    trace: DCSIIGTrace
    ledger: CommunicationLedger
    iterations: int


def _logit_gradient(coordinates, problem, logits, adjoint,
                    recommendation_floor=1e-12):
    """-F_u^T lambda, entirely in sparse block-diagonal form.

    F_u = E^T diag(-delta / (theta (q + eps))) Q with Q the softmax Jacobian, so
    F_u^T lambda = Q^T diag(...) (E lambda). Q is block-diagonal by demand group,
    hence this touches only the agent's own coordinates and never communicates.
    The dense upstream path would build a (P-G) x (P-G) matrix, which on a
    full-library instance is both enormous and unnecessary.
    """
    from brn.sue.sparse_sensitivity import (
        _recommendation_jacobian_sparse,
        sparse_reduced_basis,
    )

    q, jacobian = _recommendation_jacobian_sparse(coordinates, logits)
    basis = sparse_reduced_basis(coordinates).tocsc()
    group_by_id = {g.group_id: g for g in problem.groups}
    theta = np.asarray(
        [group_by_id[p.group_id].cost_sensitivity for p in problem.paths]
    )
    delta = np.asarray(
        [group_by_id[p.group_id].recommendation_sensitivity for p in problem.paths]
    )
    direct = -delta / (theta * (q + recommendation_floor))
    adjoint = np.asarray(adjoint, dtype=np.float64)
    block = adjoint.ndim == 2
    lifted = np.asarray(basis @ adjoint)
    if block:
        scaled = direct[:, None] * lifted
        return -np.asarray(jacobian.T @ scaled)
    return -np.asarray(jacobian.T @ (direct * lifted.ravel())).ravel()


_PAIR_CACHE: dict = {}


def _path_region_pairs(problem, decomposition) -> int:
    key = (id(problem), id(decomposition))
    if key not in _PAIR_CACHE:
        from brn.sue.operators import sparse_incidence_matrix
        D = sparse_incidence_matrix(problem).tocsc()
        region = np.asarray(decomposition.link_region)
        owner = np.asarray(decomposition.path_owner)
        count = 0
        for p in range(D.shape[1]):
            touched = set(region[D.indices[D.indptr[p]:D.indptr[p + 1]]].tolist())
            touched.discard(int(owner[p]))
            count += len(touched)
        _PAIR_CACHE[key] = count
    return _PAIR_CACHE[key]


def run_dcsiig(problem, coordinates, decomposition, options: DCSIIGOptions,
               av_flow0: FloatArray, logits0: FloatArray) -> DCSIIGResult:
    from brn.sue.equilibrium import LogitEquilibriumSolver, SolverOptions
    from brn.sue.operators import (
        bpr_link_costs, link_flows_from_paths, total_system_travel_time,
    )
    from brn.sue.projection import projected_av_step
    from .assembly import centralized_system

    solver = LogitEquilibriumSolver(
        SolverOptions(tolerance=options.follower_tolerance,
                      max_iterations=options.follower_max_iterations,
                      msa_exponent=0.7)
    )
    channels = communication_graph(decomposition, options.topology)
    compressor = build_compressor(options.compressor, seed=options.seed)
    ledger = CommunicationLedger(vector_bit_multiplier=options.relay_multiplier)

    av_flow = np.array(av_flow0, dtype=np.float64, copy=True)
    logits = np.array(logits0, dtype=np.float64, copy=True)
    owned = decomposition.coordinate_owner

    def evaluate(av, lg):
        recommendation = coordinates.recommendation_from_logits(lg)
        equilibrium = solver.solve(problem, av_path_flow=av,
                                   recommendation=recommendation)
        hdv = equilibrium.state.hdv_path_flow
        link_flow = link_flows_from_paths(problem, hdv, av)
        tstt = total_system_travel_time(
            link_flow, bpr_link_costs(problem, link_flow)
        )
        return hdv, float(tstt), equilibrium.converged

    hdv_flow, tstt, converged = evaluate(av_flow, logits)
    if not converged:
        raise RuntimeError("initial follower solve did not converge")

    trace = DCSIIGTrace()
    initial_tstt = tstt
    best = (tstt, av_flow.copy(), logits.copy())
    av_step = options.av_step
    logit_step = options.logit_step
    previous_adjoint = None

    for iteration in range(options.outer_iterations):
        if (options.communication_budget_bits is not None
                and ledger.total_bits > 0.0
                and ledger.total_bits >= options.communication_budget_bits):
            break
        systems = assemble_local_systems(
            problem, coordinates, decomposition, hdv_flow, av_flow
        )

        operator = DistributedOperator(
            matrices=[s.matrix for s in systems],
            owner=owned, touch=decomposition.coordinate_touch,
            channels=channels, compressor=compressor, ledger=ledger,
        )
        rhs = operator.aggregated_rhs([s.rhs for s in systems])
        initial = previous_adjoint if options.warm_start else None
        if options.solver == "cg_jacobi":
            preconditioner = local_jacobi_preconditioner(
                [s.matrix for s in systems], owned
            )
            solution = distributed_cg(
                operator, rhs, max_rounds=options.cg_rounds,
                tolerance=options.cg_tolerance, preconditioner=preconditioner,
                initial=initial,
            )
        else:
            from .spectral import (entropy_preconditioner, sharpened_gamma,
                                   distributed_spectral_solve)
            entropy = entropy_preconditioner(problem, hdv_flow)
            if options.solver == "cg_entropy":
                solution = distributed_cg(
                    operator, rhs, max_rounds=options.cg_rounds,
                    tolerance=options.cg_tolerance, preconditioner=entropy.apply,
                    initial=initial,
                )
            elif options.solver == "richardson":
                bound = sharpened_gamma(problem, decomposition.link_region,
                                        hdv_flow, av_flow, decomposition.n_regions)
                gamma_hat = (bound.gamma_regional
                             if options.gamma_bound == "sharp_regional"
                             else bound.gamma_path)
                # sharp_regional: one scalar per authority, once per state.
                # sharp_path: each region also sends its partial path sums to the
                # path owners (one float64 gather), then a max over authorities.
                ledger.scalar_bits += 64.0 * decomposition.n_regions
                if options.gamma_bound == "sharp_path":
                    ledger.record_vector_bits(64.0 * _path_region_pairs(
                        problem, decomposition))
                trace.gamma_hat.append(gamma_hat)
                solution = distributed_spectral_solve(
                    operator, rhs, entropy, 1.0 + gamma_hat, options.cg_rounds,
                    initial=initial, method="heavyball",
                )
            else:
                raise ValueError(options.solver)
        adjoint = solution.adjoint
        previous_adjoint = adjoint

        if (options.gradient_form == "residual"
                and getattr(solution, "residual_vector", None) is not None):
            # b - C lam = N lam + (b - A lam); N is owner-held (block diagonal by OD
            # group) and CG keeps its recursive residual r_R at the owners, so the AV
            # gradient needs no further exchange (step 5 of Algorithm 1).
            entropy_part = sum(s.matrix - s.congestion for s in systems)
            av_gradient = entropy_part @ adjoint + solution.residual_vector
        else:
            # One extra aggregation: (F_a lambda) on the coordinates each agent owns.
            congestion_operator = DistributedOperator(
                matrices=[s.congestion for s in systems],
                owner=owned, touch=decomposition.coordinate_touch,
                channels=channels, compressor=compressor, ledger=ledger,
            )
            av_gradient = rhs - congestion_operator.matvec(adjoint)
        logit_gradient = _logit_gradient(coordinates, problem, logits, adjoint)
        scale = max(float(np.max(np.abs(logit_gradient))), 1e-12)

        def candidate(av_eta, logit_eta):
            st = projected_av_step(
                coordinates, av_flow, av_gradient, av_eta, options.minimum_av_flow
            )
            lg = np.clip(logits - logit_eta * logit_gradient / scale,
                         -options.logit_bound, options.logit_bound)
            return st, lg

        if options.line_search == "armijo":
            # Restart from the initial steps at every attempt and backtrack within it
            # (follower solves only; the gradient is exchanged once per attempt).
            av_eta, logit_eta = options.av_step, options.logit_step
            accepted = None
            for _ in range(options.max_backtracks):
                step, candidate_logits = candidate(av_eta, logit_eta)
                new_hdv, new_tstt, converged = evaluate(step.flow, candidate_logits)
                trace.follower_solves += 1
                if converged and new_tstt < tstt:
                    accepted = (step, candidate_logits, new_hdv, new_tstt)
                    break
                av_eta *= options.backtrack
                logit_eta *= options.backtrack
            trace.attempt_steps.append(av_eta)
            if accepted is None:
                # Same state, same gradient: further attempts would repeat this search.
                trace.stalled = True
                break
            step, candidate_logits, new_hdv, new_tstt = accepted
            candidate_av = step.flow
            av_step = av_eta
        else:
            step, candidate_logits = candidate(av_step, logit_step)
            candidate_av = step.flow
            trace.attempt_steps.append(av_step)

            new_hdv, new_tstt, converged = evaluate(candidate_av, candidate_logits)
            trace.follower_solves += 1
            if not converged:
                av_step *= options.backtrack
                logit_step *= options.backtrack
                continue

            if new_tstt > tstt:
                # Reject and shrink: no extra follower solve is spent on a line search.
                av_step *= options.backtrack
                logit_step *= options.backtrack
                if av_step < 1e-10:
                    break
                continue

        av_flow, logits, hdv_flow, tstt = (
            candidate_av, candidate_logits, new_hdv, new_tstt
        )
        if tstt < best[0]:
            best = (tstt, av_flow.copy(), logits.copy())

        trace.tstt.append(tstt)
        trace.stationarity.append(step.normalized_mapping_norm)
        trace.adjoint_residual.append(solution.residual)
        trace.bits.append(ledger.total_bits)
        trace.av_step.append(av_step)

        if options.full_diagnostics:
            # Put the online approximate adjoint back into the full centralized
            # system at the SAME pre-update state. This diagnostic is offline and
            # excluded from the communication ledger; unlike a direct solve's
            # residual, it measures whether the protocol's answer is correct.
            diagnostic_systems = assemble_local_systems(
                problem, coordinates, decomposition, hdv_flow, av_flow
            )
            diagnostic_matrix, diagnostic_rhs = centralized_system(diagnostic_systems)
            diagnostic_residual = np.linalg.norm(
                diagnostic_matrix @ adjoint - diagnostic_rhs
            ) / max(np.linalg.norm(diagnostic_rhs), 1e-30)
            diagnostic_congestion = sum(
                system.congestion for system in diagnostic_systems
            )
            diagnostic_av = diagnostic_rhs - diagnostic_congestion @ adjoint
            diagnostic_step = projected_av_step(
                coordinates, av_flow, diagnostic_av, 1.0,
                options.minimum_av_flow,
            )
            diagnostic_logit = _logit_gradient(
                coordinates, problem, logits, adjoint
            )
            trace.full_online_adjoint_residual.append(float(diagnostic_residual))
            trace.full_online_av_mapping.append(float(diagnostic_step.normalized_mapping_norm))
            trace.full_online_logit_gradient_inf.append(float(np.max(np.abs(diagnostic_logit))))

    return DCSIIGResult(
        av_flow=best[1], logits=best[2], tstt=best[0],
        initial_tstt=initial_tstt, best_tstt=best[0],
        trace=trace, ledger=ledger, iterations=len(trace.tstt),
    )


# ---------------------------------------------------------------------------
# The regional game: heterogeneous objectives under a coalition structure
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    av_flow: FloatArray
    logits: FloatArray
    tstt: float
    initial_tstt: float
    regional: FloatArray            # J_k at the returned point
    initial_regional: FloatArray
    trace: DCSIIGTrace
    ledger: CommunicationLedger
    iterations: int
    structure: str


def run_regional_game(problem, coordinates, decomposition, structure,
                      options: DCSIIGOptions, av_flow0, logits0,
                      checkpoint=None, stop_movement: float = 0.0,
                      stop_stationarity: float = 0.0) -> GameResult:
    """Simultaneous projected gradient play among coalitions of regions.

    Every coalition descends ITS OWN objective on ITS OWN control block. The
    grand coalition uses the system-wide objective; singletons define the
    noncooperative game. Returned iterates are not certified equilibria.
    The follower equilibrium couples them all, so the
    adjoint solve is distributed no matter how selfish the agents are.

    ``checkpoint`` is a path written after EVERY accepted step. These runs are
    long and a lost process should cost one step, not the whole run.
    Steps follow ``eta_t = eta_0 / (1 + t / tau)``. A CONSTANT step does not
    converge here: measured at eta = 1, the cooperative play oscillates across
    the valley (TSTT deltas alternating +920 / -1100 while drifting down about 90
    per two steps) and the selfish play crawls at -64 per step, so after 128
    steps neither is near a fixed point. A decaying step drives raw movement to
    zero by construction, so the convergence test is ``stop_stationarity`` -- the
    normalized projected-mapping norm, which divides the step length out and is
    the actual fixed-point residual of the joint play. ``stop_movement`` remains
    only as a secondary guard.
    """
    import json as _json
    from pathlib import Path as _Path
    from brn.sue.equilibrium import LogitEquilibriumSolver, SolverOptions
    from brn.sue.operators import (
        bpr_link_costs, link_flows_from_paths, total_system_travel_time,
    )
    from brn.sue.projection import projected_av_step

    from .game import coalition_right_hand_sides, regional_road_objectives
    from .distributed import distributed_block_cg

    solver = LogitEquilibriumSolver(
        SolverOptions(tolerance=options.follower_tolerance,
                      max_iterations=options.follower_max_iterations,
                      msa_exponent=0.7)
    )
    channels = communication_graph(decomposition, options.topology)
    compressor = build_compressor(options.compressor, seed=options.seed)
    ledger = CommunicationLedger()

    owner = decomposition.coordinate_owner
    region_to_coalition = structure.region_to_coalition(decomposition.n_regions)
    coordinate_column = region_to_coalition[owner]      # which column each coordinate reads

    av_flow = np.array(av_flow0, dtype=np.float64, copy=True)
    logits = np.array(logits0, dtype=np.float64, copy=True)

    def evaluate(av, lg):
        recommendation = coordinates.recommendation_from_logits(lg)
        equilibrium = solver.solve(problem, av_path_flow=av,
                                   recommendation=recommendation)
        hdv = equilibrium.state.hdv_path_flow
        link_flow = link_flows_from_paths(problem, hdv, av)
        tstt = total_system_travel_time(
            link_flow, bpr_link_costs(problem, link_flow)
        )
        return hdv, float(tstt), equilibrium.converged

    hdv_flow, tstt, converged = evaluate(av_flow, logits)
    if not converged:
        raise RuntimeError("initial follower solve did not converge")
    initial_regional = regional_road_objectives(
        problem, decomposition.link_region, hdv_flow, av_flow,
        decomposition.n_regions
    )

    trace = DCSIIGTrace()
    initial_tstt = tstt
    best = (tstt, av_flow.copy(), logits.copy(), hdv_flow.copy())
    av_step, logit_step = options.av_step, options.logit_step
    previous_block = None

    for iteration in range(options.outer_iterations):
        if options.step_decay_tau > 0.0:
            decay = 1.0 / (1.0 + iteration / options.step_decay_tau)
            av_step = options.av_step * decay
            logit_step = options.logit_step * decay
        systems = assemble_local_systems(
            problem, coordinates, decomposition, hdv_flow, av_flow
        )

        operator = DistributedOperator(
            matrices=[s.matrix for s in systems], owner=owner,
            touch=decomposition.coordinate_touch, channels=channels,
            compressor=compressor, ledger=ledger,
        )
        if options.count_rhs:
            # v2: each region's objective derivative lives on coordinates owned by
            # other authorities; delivering it is a counted (and maskable) gather.
            pieces = []
            for region, system in enumerate(systems):
                piece = np.zeros((owner.size, structure.n_coalitions))
                piece[:, region_to_coalition[region]] = system.rhs
                pieces.append(piece)
            rhs_block = operator.aggregated_rhs(pieces)
        else:
            rhs_block = coalition_right_hand_sides(systems, structure)
        initial = previous_block if options.warm_start else None
        if options.solver == "cg_entropy":
            from .spectral import entropy_preconditioner
            preconditioner = entropy_preconditioner(problem, hdv_flow).apply
        else:
            preconditioner = local_jacobi_preconditioner(
                [s.matrix for s in systems], owner
            )
        solution = distributed_block_cg(
            operator, rhs_block, max_rounds=options.cg_rounds,
            tolerance=options.cg_tolerance, preconditioner=preconditioner,
            initial=initial,
        )
        previous_block = solution.adjoint

        congestion_operator = DistributedOperator(
            matrices=[s.congestion for s in systems], owner=owner,
            touch=decomposition.coordinate_touch, channels=channels,
            compressor=compressor, ledger=ledger,
        )
        av_block = rhs_block - congestion_operator.matvec(solution.adjoint)

        rows = np.arange(owner.size)
        av_gradient = av_block[rows, coordinate_column]
        # One sparse call for every coalition at once: rebuilding the softmax
        # Jacobian per coalition dominated the iteration cost.
        logit_columns = _logit_gradient(
            coordinates, problem, logits, solution.adjoint
        )
        logit_gradient = logit_columns[rows, coordinate_column]

        step = projected_av_step(
            coordinates, av_flow, av_gradient, av_step, options.minimum_av_flow
        )
        scale = max(float(np.max(np.abs(logit_gradient))), 1e-12)
        candidate_logits = np.clip(
            logits - logit_step * logit_gradient / scale,
            -options.logit_bound, options.logit_bound,
        )

        new_hdv, new_tstt, converged = evaluate(step.flow, candidate_logits)
        if not converged:
            av_step *= options.backtrack
            logit_step *= options.backtrack
            continue

        # In a game there is no shared objective to monotonically decrease, so
        # acceptance is by the fixed-point residual of the joint play, not by
        # TSTT: insisting on TSTT descent would silently re-impose cooperation.
        moved = float(np.linalg.norm(step.flow - av_flow)) + float(
            np.linalg.norm(candidate_logits - logits)
        )
        av_flow, logits, hdv_flow, tstt = (
            step.flow, candidate_logits, new_hdv, new_tstt
        )
        if tstt < best[0]:
            best = (tstt, av_flow.copy(), logits.copy(), hdv_flow.copy())

        trace.tstt.append(tstt)
        trace.stationarity.append(step.normalized_mapping_norm)
        trace.adjoint_residual.append(solution.residual)
        trace.bits.append(ledger.total_bits)
        trace.av_step.append(av_step)
        trace.movement.append(moved)


        if checkpoint is not None:
            path = _Path(checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps({
                **result_metadata("regional_game_checkpoint"),
                "structure": structure.label,
                "step": len(trace.tstt),
                "tstt": tstt,
                "initial_tstt": initial_tstt,
                "movement": moved,
                "stationarity": step.normalized_mapping_norm,
                "bits": ledger.total_bits,
                "av_step": av_step,
                "tstt_trace": trace.tstt,
                "movement_trace": trace.movement,
            }))

        if step.normalized_mapping_norm < stop_stationarity:
            break
        if moved < max(stop_movement, 1e-12):
            break

    final_regional = regional_road_objectives(
        problem, decomposition.link_region, hdv_flow, av_flow,
        decomposition.n_regions
    )
    return GameResult(
        av_flow=av_flow, logits=logits, tstt=tstt, initial_tstt=initial_tstt,
        regional=final_regional, initial_regional=initial_regional,
        trace=trace, ledger=ledger, iterations=len(trace.tstt),
        structure=structure.label,
    )
