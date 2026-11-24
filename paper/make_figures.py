"""Figures 1-5 of the paper from the run records.

usage: python paper/make_figures.py [OUT_DIR]        (default: figures/)

Figures 1, 2 and 4 use the revision records (residual-form AV gradient, masked protocols on
the common ledger: results/revision/); Figures 3 and 5 use the submission records
(results/submission/). Colour roles are fixed across figures: neighbor-only exchange =
vermillion, EPDA = blue, state broadcast = gold, coordinator = gray, reference = ink.
Coverage levels use a single-hue blue ramp (light = low coverage).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/submission"
REV = ROOT / "results/revision"
OUT = ROOT / "figures"
BLUE, RED, TEAL, GRAY, GOLD = "#0072B2", "#D55E00", "#009E73", "#747C82", "#BB8614"
INK, MUTED, GRID = "#24292D", "#5F6870", "#E3E7EA"
RAMP = ["#BFD7EA", "#7FB0D6", "#3F86BF", "#0B4F8A"]          # coverage 25/50/75/100 %
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.0,
                     "axes.labelsize": 7.4, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#90989F", "legend.fontsize": 6.6,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def jl_submission(name):
    p = R / name
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def jl(name):
    """JSONL record: residual-form revision probe if one exists, else the submission run."""
    p = REV / "figs" / name
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else jl_submission(name)


def js(name):
    """JSON record: residual-form revision probe if one exists, else the submission run."""
    for p in (REV / "figs" / name, R / name):
        if p.exists():
            return json.load(open(p))
    return None


def rec(rows, config):
    return next((r for r in rows if r.get("config") == config), None)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- outcomes by load (Table 1)
SCEN = [("nominal", "risk_base_Anaheim_K16.jsonl", "risk_base_state_Anaheim_K16.jsonl", None),
        ("drivers\n4x more\nsensitive", "risk_theta032_Anaheim_K16.jsonl", "risk_theta032_state_Anaheim_K16.jsonl",
         "risk_theta032_epda_Anaheim_K16.jsonl"),
        ("demand x2,\nsensitivity x4", "risk_high_Anaheim_K16.jsonl", "risk_high_state_Anaheim_K16.jsonl", None)]


def outcomes():
    out = []
    for (lab, f_main, f_state, f_epda), g in zip(SCEN, ("base", "theta032", "high")):
        rows = []
        for suffix in ("a", "b", "r4"):
            p = REV / f"t1_{g}_{suffix}.jsonl"
            if p.exists():
                rows += [json.loads(l) for l in open(p) if l.strip()]
        t = {r["config"]: r for r in rows}
        ex = t["exactN"]; base = 1 - ex["tstt"] / ex["initial_tstt"]
        r = lambda x: None if x is None else 100 * (1 - x["tstt"] / x["initial_tstt"]) / base
        mb = lambda x: None if x is None else x["total_bits"] / 1e6
        road = t.get("roadN@1e-6+warm+res"); ep2 = t.get("N:q8+r2+warm+res"); ep4 = t.get("N:q8+r4+warm+res")
        s = jl_submission(f_state); bc = rec(s, "allgather-uniform8")
        bcr = None if bc is None else 100 * bc["tstt_improvement"] / rec(jl_submission(f_main), "exact")["tstt_improvement"]
        out.append(dict(label=lab, exact=100 * base, road=r(road), road_mbit=mb(road), road_acc=road and road.get("accepted_steps"),
                        epda2=r(ep2), epda2_mbit=mb(ep2), epda4=r(ep4) if g == "high" else None, epda4_mbit=mb(ep4) if g == "high" else None,
                        bcast=bcr, bcast_mbit=None if bc is None else bc["total_bits"] / 1e6))
    return out


def schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
    cols = ["#FBE9E1", "#EEF3EE", "#EAF1F8"]
    y0, h = 2.55, 4.45
    for i in range(3):
        ax.add_patch(Rectangle((0.1 + 3.3 * i, y0), 3.2, h, fc=cols[i], ec="none", zorder=0))
        ax.text(1.7 + 3.3 * i, y0 + h - 0.4, "Region " + "ABC"[i], ha="center", va="center", fontsize=6.8, weight="bold")
    fire = Polygon([(0.35, 2.85), (0.85, 4.0), (0.75, 4.7), (1.25, 4.2), (1.45, 5.3), (1.85, 4.15), (2.15, 2.85)],
                   closed=True, fc="#F2B79B", ec=RED, lw=0.8, zorder=1)
    ax.add_patch(fire); ax.text(1.25, 3.1, "fire", ha="center", fontsize=6.0, color=RED, zorder=3)
    north = np.array([(2.2, 4.3), (2.9, 5.55), (6.6, 5.55), (8.7, 4.6)])
    south = np.array([(2.2, 4.3), (2.9, 3.25), (6.6, 3.25), (8.7, 4.0)])
    ax.plot(north[:, 0], north[:, 1], color=BLUE, lw=1.6, zorder=2)
    ax.plot(south[:, 0], south[:, 1], color=TEAL, lw=1.6, ls=(0, (3, 1.6)), zorder=2)
    ax.add_patch(Circle((8.95, 4.3), 0.26, fc="white", ec=INK, lw=0.9, zorder=3))
    ax.text(8.95, 3.75, "shelter", ha="center", va="top", fontsize=5.8, color=MUTED)
    ax.text(4.75, 5.7, "AVs + guidance", ha="center", va="bottom", fontsize=5.9, color=BLUE)
    ax.text(4.75, 3.12, "drivers re-equilibrate", ha="center", va="top", fontsize=5.9, color=TEAL)
    xs = [1.7, 5.0, 8.3]
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, 0.75), 0.4, fc="white", ec=INK, lw=0.9, zorder=3))
        ax.text(x, 0.75, "ABC"[i], ha="center", va="center", fontsize=6.8, weight="bold", zorder=4)
    for a, b in ((0, 1), (1, 2)):
        ax.plot([xs[a] + 0.45, xs[b] - 0.45], [0.75, 0.75], color=GRAY, lw=1.4, zorder=2)
    ax.text(3.35, 0.55, "road channel", ha="center", va="top", fontsize=5.4, color=MUTED)
    arr = FancyArrowPatch((xs[2] - 0.1, 1.2), (xs[0] + 0.1, 1.2), connectionstyle="arc3,rad=0.22",
                          arrowstyle="-|>", mutation_scale=7, color=RED, lw=1.1, ls=(0, (3, 1.5)), zorder=2)
    ax.add_patch(arr)
    ax.text(5.0, 2.05, "dependency A needs from C", ha="center", va="center", fontsize=5.8, color=RED,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_title("(a) evacuation agents", loc="left")


def fig1():
    tr = js("risk_trace_Anaheim_K16.json"); oc = outcomes()
    fig = plt.figure(figsize=(7.0, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 1.08], wspace=0.34)
    schematic(fig.add_subplot(gs[0]))
    ax = fig.add_subplot(gs[1])
    if tr:
        rd = [r for r in tr if r["protocol"] == "road"]; cp = [r for r in tr if r["protocol"] == "complete"]
        x = [r["rounds"] for r in rd]
        ax.semilogy(x, [max(r["internal_residual"], 1e-16) for r in rd], color=RED, lw=1.2, ls=(0, (3, 1.5)))
        ax.semilogy(x, [r["grad_err"] for r in rd], color=RED, lw=1.8)
        xc = [r["rounds"] for r in cp]
        ax.semilogy(xc, [r["grad_err"] for r in cp], color=BLUE, lw=1.8, marker="o", ms=2.6)
        ax.text(40, rd[-1]["grad_err"] * 1.6, "neighbor-only: gradient error", ha="right", va="bottom", fontsize=6.0, color=RED)
        ax.text(40, 3e-15, "its own residual", ha="right", va="bottom", fontsize=6.0, color=RED)
        ax.text(9, cp[-1]["grad_err"] * 0.45, "complete, 8 bit:\ngradient error", ha="left", va="top", fontsize=6.0, color=BLUE)
        ax.set_ylim(1e-16, 20); ax.set_xlim(0, 41)
        ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
    ax.set_xlabel("communication rounds"); ax.set_ylabel("relative error")
    ax.grid(color=GRID, lw=0.6); ax.set_title("(b) converged ...", loc="left")
    ax = fig.add_subplot(gs[2])
    n = len(oc); xb = np.arange(n); w = 0.26
    def val(o, k):
        return o[k] if o[k] is not None else np.nan
    road = [val(o, "road") for o in oc]
    ep = [val(o, "epda4") if o["epda4"] is not None and o is oc[-1] else val(o, "epda2") for o in oc]
    bc = [val(o, "bcast") for o in oc]
    ax.bar(xb - w, road, w * 0.92, color=RED, label="neighbor-only (64 b)")
    ax.bar(xb, ep, w * 0.92, color=BLUE, label="EPDA (8 b)")
    ax.bar(xb + w, bc, w * 0.92, color=GOLD, label="broadcast (8 b)")
    for i, o in enumerate(oc):
        if o["road_mbit"]:
            ax.text(i - w, road[i] + 2, f"{o['road_mbit']:.0f}", ha="center", va="bottom", fontsize=5.4, color=MUTED)
        m = o["epda4_mbit"] if i == n - 1 and o["epda4"] is not None else o["epda2_mbit"]
        if m:
            ax.text(i, ep[i] + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=5.4, color=MUTED)
        if o["bcast_mbit"]:
            ax.text(i + w, bc[i] + 2, f"{o['bcast_mbit']:.0f}", ha="center", va="bottom", fontsize=5.4, color=MUTED)
    ax.set_xticks(xb, [o["label"] for o in oc], fontsize=5.9)
    ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100]); ax.axhline(100, color=INK, lw=0.6, ls=":")
    ax.set_ylabel("share of attainable reduction (%)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_title("(c) ... but misdirected", loc="left")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.66, -0.01), ncol=3, fontsize=6.0,
               handlelength=1.0, columnspacing=1.2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.31)
    save(fig, "fig1_teaser")
    return oc


# ---------------------------------------------------------------- Figure: coverage vs precision
def fig_coverage():
    st = js("risk_story_checks_Anaheim_K16.json"); tr = js("risk_trace_Anaheim_K16.json")
    if not st:
        return
    rows = st["c_rows"]
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    bits = {"uniform4": "4", "uniform6": "6", "uniform8": "8", "uniform16": "16", "float64": "64"}
    for c, col in zip((0.25, 0.5, 0.75, 1.0), RAMP):
        S = [r for r in rows if r["coverage"] == c]
        ax.loglog([r["mbit"] for r in S], [r["grad_err"] for r in S], "-o", color=col, ms=3.2, lw=1.4,
                  label=f"{int(c * 100)}% of dependencies")
        if c == 1.0:
            for r in S:
                if r["compressor"] in ("uniform4", "uniform8", "float64"):
                    ax.annotate(f"{bits[r['compressor']]} bit", (r["mbit"], r["grad_err"]), textcoords="offset points",
                                xytext=(-2, -9), ha="center", fontsize=5.6, color=MUTED)
    if tr:
        rd = [r for r in tr if r["protocol"] == "road" and r["rounds"] == 2][0]
        ax.loglog([rd["mbit"]], [rd["grad_err"]], "X", color=RED, ms=6.5, label="neighbor-only, 64 bit")
    ax.set_xlabel("Mbit per gradient (two rounds)"); ax.set_ylabel("relative gradient error")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    ax.grid(color=GRID, lw=0.6, which="major"); ax.set_ylim(1.5e-3, 1.5)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", bbox_to_anchor=(1.0, 0.13), handlelength=1.6)
    fig.tight_layout(pad=0.3)
    save(fig, "fig2_coverage")


# ---------------------------------------------------------------- Figure: where the missing information sits
def fig_where():
    loc = jl("risk_locality_Anaheim.jsonl"); gs = js("risk_gamma_share.json")
    if not loc and not gs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9), gridspec_kw={"width_ratios": [1, 1.12]})
    ax = axs[0]
    if loc:
        Ks = sorted({r["K"] for r in loc}); beyond = []
        for K in Ks:
            S = [r for r in loc if r["K"] == K]; tot = sum(r["infl_eq"] ** 2 for r in S)
            beyond.append(sum(r["infl_eq"] ** 2 for r in S if r["hops"] > 1) / tot)
        x = np.arange(len(Ks)); ax.bar(x, beyond, color=RED, width=0.6)
        for i, b in enumerate(beyond):
            ax.text(i, b + 0.02, f"{b:.0%}", ha="center", fontsize=6.2)
        ax.set_xticks(x, [str(k) for k in Ks]); ax.set_ylim(0, 1)
        ax.set_xlabel("authorities K"); ax.set_ylabel("influence beyond one road hop")
        ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("(a) locality", loc="left")
    ax = axs[1]
    if gs:
        g = sorted(gs, key=lambda r: r["gamma_exact"])
        gx = [r["gamma_exact"] for r in g]
        ax.semilogx(gx, [r["response_share"] for r in g], "o", color=TEAL, ms=3.6, label="measured")
        xx = np.logspace(np.log10(min(gx) * 0.7), np.log10(max(gx) * 1.4), 100)
        ax.semilogx(xx, xx / (1 + xx), color=INK, lw=1.0, ls=(0, (3, 1.5)), label="bound (Prop. 5)")
        ax.set_ylim(0, 1); ax.set_xlabel(r"exact spread $\gamma^\star$")
        ax.set_ylabel("response share of adjoint")
        ax.legend(frameon=False, loc="upper left", fontsize=6.0, borderaxespad=0.2)
        ax.grid(color=GRID, lw=0.6, which="both")
    ax.set_title("(b) driver response", loc="left")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "fig3_where")


# ---------------------------------------------------------------- Figure: protocol families
def fig_protocols():
    ss = {r["protocol"]: r for r in jl("risk_state_sharing_Anaheim_K16.jsonl")}
    tr = js("risk_trace_Anaheim_K16.json")
    if not ss:
        return
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    fm = ["uniform6", "uniform8", "uniform10", "uniform12", "uniform16"]
    for prefix, col, lab in (("coord", GRAY, "coordinator"), ("allgather", GOLD, "broadcast")):
        p = np.array([(ss[f"{prefix}-{f}"]["mbit"], ss[f"{prefix}-{f}"]["grad_err"]) for f in fm if f"{prefix}-{f}" in ss])
        ax.loglog(p[:, 0], p[:, 1], "-s", color=col, ms=2.8, lw=1.1, label=lab)
    ep = np.array([(ss[k]["mbit"], ss[k]["grad_err"]) for k in ("EPDA-CG q8 R1", "EPDA-CG q8 R2", "EPDA-CG q8 R3") if k in ss])
    ax.loglog(ep[:, 0], ep[:, 1], "-o", color=BLUE, ms=3, lw=1.3, label="EPDA, 8 bit")
    if tr:
        rd = [r for r in tr if r["protocol"] == "road" and r["rounds"] == 10][0]
        ax.loglog([rd["mbit"]], [rd["grad_err"]], "X", color=RED, ms=6.5, label="neighbor-only")
    ax.set_xlim(0.05, 14); ax.set_xlabel("Mbit per gradient (K=16)"); ax.set_ylabel("relative gradient error")
    ax.grid(color=GRID, lw=0.6, which="both"); ax.set_title("(a) accuracy vs payload", loc="left")
    ax = axs[1]
    Ks = [4, 8, 16, 32, 64]; pts = {"coord": [], "allgather": [], "epda": []}
    for K in Ks:
        S = {r["protocol"]: r for r in jl(f"risk_state_sharing_Anaheim_K{K}.jsonl")}
        if not S:
            continue
        e = S.get("EPDA-CG q8 R2"); target = e["grad_err"] if e else None
        def match(prefix):
            c = [S[f"{prefix}-{f}"] for f in fm if f"{prefix}-{f}" in S and S[f"{prefix}-{f}"]["grad_err"] <= max(target, 1e-9) * 1.25]
            return min((r["mbit"] for r in c), default=np.nan)
        pts["coord"].append((K, match("coord"))); pts["allgather"].append((K, match("allgather")))
        pts["epda"].append((K, e["mbit"]))
    for key, col, mk in (("coord", GRAY, "s"), ("allgather", GOLD, "s"), ("epda", BLUE, "o")):
        if pts[key]:
            p = np.array(pts[key]); ax.semilogy(p[:, 0], p[:, 1], "-" + mk, color=col, ms=3, lw=1.2)
    ax.set_xscale("log", base=2); ax.set_xticks(Ks, [str(k) for k in Ks])
    ax.set_xlabel("authorities K"); ax.set_ylabel("Mbit per gradient")
    ax.grid(color=GRID, lw=0.6, which="both"); ax.set_title("(b) scaling", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", fontsize=6.0, handlelength=1.4, columnspacing=0.9)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.1, 1, 1))
    save(fig, "fig4_protocols")


# ---------------------------------------------------------------- Figure: failures
def fig_failures():
    F = jl("risk_failures_Anaheim_K16.jsonl")
    if not F:
        return
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85))
    ax = axs[0]
    ps = sorted({r["p"] for r in F})
    for rule, col, lab in (("relay", BLUE, "relay over survivors"), ("neighbour", RED, "neighbor-only")):
        med = []; lo = []; hi = []
        for p in ps:
            e = [r["err_2"] for r in F if r["p"] == p and r["rule"] == rule]
            med.append(np.median(e)); lo.append(np.percentile(e, 25)); hi.append(np.percentile(e, 75))
        ax.plot(ps, med, "-o", color=col, ms=2.8, lw=1.3, label=lab)
        ax.fill_between(ps, lo, hi, color=col, alpha=0.18, lw=0)
    part = [np.mean([r["components"] > 1 for r in F if r["p"] == p and r["rule"] == "relay"]) for p in ps]
    ax.plot(ps, part, ":", color=INK, lw=1.0, label="P(partitioned)")
    ax.set_xlabel("failure probability $p$"); ax.set_ylabel("adjoint error")
    ax.set_ylim(-0.03, 1.25); ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", ncol=1, handlelength=1.4, borderaxespad=0.1)
    ax.set_title("(a) error under failures", loc="left")
    ax = axs[1]
    P = [r for r in F if r["rule"] == "relay" and r["components"] > 1]
    if P:
        x = [1 - r.get("coverage", 1.0) for r in P]
        ax.plot(x, [r["err_N"] for r in P], "o", color=BLUE, ms=2.8, label="relay")
        ax.plot(x, [r["flux"] for r in P], "_", color=INK, ms=6, label="flux bound")
        ax.set_xlabel("undelivered share of $G_{\\rm dep}$"); ax.set_ylabel("adjoint error ($N$-norm)")
        ax.grid(color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=5.6)
    ax.set_title("(b) partitioned draws", loc="left")
    fig.tight_layout(pad=0.6, w_pad=0.8)
    save(fig, "fig5_failures")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for o in fig1():
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items()})
    fig_coverage(); fig_where(); fig_protocols(); fig_failures()
