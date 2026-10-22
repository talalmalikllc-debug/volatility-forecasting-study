#!/usr/bin/env python3
"""Build every figure in figures/ from the JSON results in results/.

Runs without the broker export.

    python figures/make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
SYMS = ["US100.cash", "US500.cash", "US30.cash", "GER40.cash", "XAUUSD", "EURUSD"]
LABEL = {"US100.cash": "US100 (NQ)", "US500.cash": "US500 (ES)", "US30.cash": "US30 (YM)",
         "GER40.cash": "GER40 (FDAX)", "XAUUSD": "XAUUSD (Gold)", "EURUSD": "EURUSD"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def load(name):
    return json.load(open(RES / name))


def bh(p, q=0.10):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    out = np.zeros(m, bool)
    out[o[:k]] = True
    return out


def fig_persistence():
    r1 = load("r1_persistence.json")
    ks = sorted(int(k) for k in r1["US100.cash"]["adj"]["k"])
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, s in zip(axs.flat, SYMS):
        d = r1[s]["adj"]
        ps = [d["k"][str(k)]["per_state"]["compressed"] for k in ks]
        obs = [p["P_same"] for p in ps]
        lo = [p["ci"][0] for p in ps]
        hi = [p["ci"][1] for p in ps]
        mk = [p["markov_k_step"] for p in ps]
        unc = d["unconditional"]["compressed"]
        ax.fill_between(ks, lo, hi, color=BLUE, alpha=0.2, lw=0)
        ax.plot(ks, obs, color=BLUE, lw=2, marker="o", ms=5, label="observed")
        ax.plot(ks, mk, color=ORANGE, lw=1.6, ls="--", label="first-order Markov $P^k$")
        ax.axhline(unc, color=GRAY, lw=1.4, ls=":", label="unconditional rate")
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_title(LABEL[s])
    for ax in axs[1]:
        ax.set_xlabel("horizon k (5-minute bars)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$P(\mathrm{compressed}_{t+k}\mid\mathrm{compressed}_t)$")
    axs[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Compression is a persistent regime: it outlives what a memoryless Markov chain predicts",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_regime_persistence.png", dpi=180)
    plt.close(fig)


def fig_fdr():
    ents = load("research_log.json")["entries"]
    pre = [e for e in ents if e["prespecified"] and e.get("p_value") is not None]
    fam = {
        "volatility family (102 tests)": [e for e in pre if e["part"] in ("5", "11/12", "4", "6", "2")],
        "directional family (132 tests)": [e for e in pre if e["part"] in ("7", "8/9/10")],
    }
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (name, es), c in zip(axs, fam.items(), (BLUE, ORANGE)):
        p = np.sort(np.array([e["p_value"] for e in es]))
        m = len(p)
        rank = np.arange(1, m + 1)
        surv = bh(p)
        pp = np.maximum(p, 1e-4)
        ax.scatter(rank[~surv], pp[~surv], s=16, color=GRAY, zorder=2, label="rejected by BH")
        ax.scatter(rank[surv], pp[surv], s=16, color=c, zorder=3, label=f"survives BH q = 0.10 ({surv.sum()})")
        ax.plot(rank, 0.10 * rank / m, color=INK, lw=1.3, label=r"BH line $q\,i/m$")
        ax.axhline(0.05, color=INK2, lw=0.9, ls=":", label=f"raw p = 0.05 ({(p < 0.05).sum()} below)")
        ax.set_yscale("log")
        ax.set_ylim(8e-5, 1.3)
        ax.set_xlabel("rank i of sorted p-value")
        ax.set_title(name)
        ax.legend(loc="lower right", fontsize=8)
    axs[0].set_ylabel("p-value (floored at 1e-4 for display)")
    fig.suptitle("Benjamini–Hochberg control across the whole pre-registered test tree",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_fdr_by_family.png", dpi=180)
    plt.close(fig)


def fig_sessions():
    r4 = load("r4_sessions.json")
    bnds = ["asia_to_london", "london_to_ny", "ny_cash_open", "ny_am_to_pm"]
    blab = ["Asia → London", "London → NY", "NY cash open", "NY AM → PM"]
    fig, axs = plt.subplots(1, 4, figsize=(12, 4.2), sharey=True)
    y = np.arange(len(SYMS))[::-1]
    for ax, b, bl in zip(axs, bnds, blab):
        for yi, s in zip(y, SYMS):
            d = r4[s][b]
            q, r = d["P_spike_quiet"], d["P_spike_rest"]
            ax.plot([q, r], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.scatter(q, yi, color=BLUE, s=46, zorder=3)
            ax.scatter(r, yi, color=ORANGE, s=46, zorder=3)
            ax.text((q + r) / 2, yi + 0.22, f"{100 * d['spike_diff']['diff']:+.0f} pp",
                    ha="center", fontsize=7.5, color=INK2)
        ax.set_title(bl)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("P(volatility spike in next 2h)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, [LABEL[s] for s in SYMS])
    axs[0].scatter([], [], color=BLUE, label="quiet run-in (bottom quintile)")
    axs[0].scatter([], [], color=ORANGE, label="all other days")
    fig.legend(loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Volatility scale is forecastable: a quiet run-in predicts a quiet handover, 24/24 cells",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig3_session_handover_forecast.png", dpi=180)
    plt.close(fig)


def fig_directional():
    r5 = load("r5_directional.json")
    ks = sorted(int(k) for k in r5["US100.cash"]["part7"])
    fig, ax = plt.subplots(figsize=(9, 5.0))
    w = 0.12
    cols = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#6250d6"]
    for i, (s, c) in enumerate(zip(SYMS, cols)):
        x = np.arange(len(ks)) + (i - 2.5) * w
        d = [r5[s]["part7"][str(k)]["signed"] for k in ks]
        m = np.array([v["mean"] for v in d])
        lo = np.array([v["ci_lo"] for v in d])
        hi = np.array([v["ci_hi"] for v in d])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", ms=5, color=c, capsize=2, lw=1.1, label=LABEL[s])
    sp = np.mean([r5[s]["mean_spread_R"] for s in SYMS])
    ax.axhspan(-sp, sp, color=GRAY, alpha=0.18, lw=0, label=f"± mean spread at entry ({sp:.3f} R)")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xticks(range(len(ks)), [f"{k} bars\n({5 * k} min)" for k in ks])
    ax.set_ylabel("signed return, units of impulse-bar range R")
    ax.set_title("No directional edge after an impulse out of compression (day-block 95% CI)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(FIG / "fig4_directional_null.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_persistence()
    fig_fdr()
    fig_sessions()
    fig_directional()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
