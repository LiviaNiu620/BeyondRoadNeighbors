"""Exposure-weighted objective J = sum_a w_a v_a t_a(v_a), w_a = 1 + rho * exp(-hops_a / ell).

The fire zone is the K=16 spectral region with the largest originating demand at nominal
demand; hops_a is the road-graph distance (in links) from the fire zone (0 inside it).
Only the objective changes: the right-hand side b and the acceptance test use w; the
follower equilibrium, the matrix A, the dependency structure and gamma are untouched.

Usage (before importing any experiment script):  import risk; risk.apply(rho=2.0)
"""
import json
from collections import deque
from pathlib import Path
import numpy as np
WEIGHTS = None
INFO = {}


def build_weights(rho=2.0, ell=2.0, K=None, network=None, fire=None):
    import os
    from brn.experiments.common import load, regions
    from brn.experiments import demand as scaled
    network = network or os.environ.get("RISK_NETWORK", "Anaheim")
    K = K or int(os.environ.get("RISK_K", "16"))
    scaled.apply(1.0)
    problem, coords, av, lg = load(network)
    dec = regions(problem, coords, K)
    owner = np.asarray(dec.path_owner)
    demand = np.zeros(K)
    gid = {g.group_id: g for g in problem.groups}
    seen = set()
    for p_idx, p in enumerate(problem.paths):
        if p.group_id in seen:
            continue
        seen.add(p.group_id)
        g = gid[p.group_id]
        demand[owner[p_idx]] += g.hdv_demand + g.av_demand if hasattr(g, "hdv_demand") else 0.0
    if demand.sum() == 0:  # fall back to counting paths per owner
        demand = np.bincount(owner, minlength=K).astype(float)
    if fire is None and os.environ.get("RISK_FIRE"):
        fire = int(os.environ["RISK_FIRE"])
    if fire is None:
        fire = int(np.argmax(demand))
    rank = int(np.sum(demand > demand[fire])) + 1
    lreg = np.asarray(dec.link_region)
    adj = {}
    for a, link in enumerate(problem.links):
        adj.setdefault(link.tail, set()).add(link.head); adj.setdefault(link.head, set()).add(link.tail)
    src = {problem.links[a].tail for a in np.where(lreg == fire)[0]} | {problem.links[a].head for a in np.where(lreg == fire)[0]}
    dist = {n: 0 for n in src}; q = deque(src)
    while q:
        u = q.popleft()
        for w in adj.get(u, ()):
            if w not in dist:
                dist[w] = dist[u] + 1; q.append(w)
    hops = np.array([0 if lreg[a] == fire else min(dist.get(l.tail, 99), dist.get(l.head, 99)) for a, l in enumerate(problem.links)], float)
    w = 1.0 + rho * np.exp(-hops / ell)
    info = dict(network=network, rho=rho, ell=ell, K=K, fire_region=fire, fire_demand_rank=rank, fire_links=int(np.sum(lreg == fire)),
                links_w_over_1_5=int(np.sum(w > 1.5)), mean_w=float(w.mean()), max_w=float(w.max()))
    return w, info


def apply(rho=2.0, ell=2.0):
    global WEIGHTS, INFO
    WEIGHTS, INFO = build_weights(rho, ell)
    w = WEIGHTS
    import brn.sue.operators as ops
    import brn, brn.assembly as asm, brn.outer as outer
    from dataclasses import replace
    orig_assemble = asm.assemble_local_systems

    def weighted_tstt(link_flow, link_cost):
        return float(np.sum(w * np.asarray(link_flow) * np.asarray(link_cost)))

    def weighted_assemble(problem, coordinates, decomposition, hdv_path_flow, av_path_flow):
        systems = orig_assemble(problem, coordinates, decomposition, hdv_path_flow, av_path_flow)
        from brn.sue.operators import bpr_link_costs, bpr_link_derivatives, link_flows_from_paths, sparse_incidence_matrix
        from brn.sue.sparse_sensitivity import sparse_reduced_basis
        inc = sparse_incidence_matrix(problem).tocsr(); basis = sparse_reduced_basis(coordinates).tocsc()
        v = link_flows_from_paths(problem, hdv_path_flow, av_path_flow)
        m = w * (bpr_link_costs(problem, v) + v * bpr_link_derivatives(problem, v))
        out = []
        for s in systems:
            links = np.where(decomposition.link_region == s.region)[0]
            rhs = np.asarray(basis.T @ (inc[links, :].T @ m[links])).ravel()
            out.append(replace(s, rhs=rhs))
        return out

    ops.total_system_travel_time = weighted_tstt
    asm.assemble_local_systems = weighted_assemble
    outer.assemble_local_systems = weighted_assemble
    brn.assemble_local_systems = weighted_assemble
    from brn.experiments import common
    common.assemble_local_systems = weighted_assemble
    try:
        from brn.experiments import state_sharing as ss
        orig_build = ss.StateSystem.build

        def build(self, v, f):
            A, b, C = orig_build(self, v, f)
            from brn.sue.operators import bpr_link_costs, bpr_link_derivatives
            m = w * (bpr_link_costs(self.problem, v) + v * bpr_link_derivatives(self.problem, v))
            return A, np.asarray(self.DE.T @ m).ravel(), C
        ss.StateSystem.build = build
    except ImportError:
        pass
    print("risk objective:", INFO, flush=True)
    return INFO
