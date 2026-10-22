"""R3 -- Parts 3, 4, 6: depth, duration, and raw-vs-clock-standardised.

Everything here is a MATCHED comparison. Bucketing episodes by depth or duration
and reading off raw probabilities is not enough, because long or deep episodes
do not occur uniformly across the day or across the sample: comparing bucket to
bucket then compares clock and regime composition as much as compression. Every
number below is therefore a lift against ordinary bars in the same five-minute
slot of the same calendar month.

Outcomes (both measured over the 24 bars following the conditioning bar, and
both referenced to the seasonal profile so "expansion" is a claim about the
market and not about what time it is):

    spike    max over the window of (bar range / seasonal range) >= 2.0
    fwd_abn  sum(range) / sum(seasonal range)

Part 6 is the head-to-head the whole study turns on: given the same outcome and
the same matching, does ABNORMAL compression (quiet for this clock time) carry
more information than RAW compression (low ATR outright)?
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import research_log as RL
import stats_util as S
import vol_core as V

OUT = Path(_os.path.join(_REPO, "results"))
SYMS = ["XAUUSD", "EURUSD", "US100.cash", "US500.cash", "US30.cash", "GER40.cash"]
W = 24
SPIKE_ABN = 2.0


def forward_measures(f):
    R = f["R"].to_numpy()
    seas = f["atr_seasonal"].to_numpy()
    gap = f["gap"].to_numpy()
    n = len(f)
    rw = sliding_window_view(R, W)
    sw = sliding_window_view(seas, W)
    m = n - W - 1
    ok = ~np.isnan(sw[1:m + 1]).any(axis=1)
    fwd = np.full(n, np.nan); peak = np.full(n, np.nan)
    fwd[:m] = np.where(ok, rw[1:m + 1].sum(1) / sw[1:m + 1].sum(1), np.nan)
    peak[:m] = np.where(ok, (rw[1:m + 1] / sw[1:m + 1]).max(1), np.nan)
    valid = np.zeros(n, bool)
    valid[:m] = ok & V.contiguous_forward(gap, W)[:m]
    valid &= ~np.isnan(f["abn"].to_numpy()) & ~np.isnan(f["pct_raw"].to_numpy())
    return fwd, peak, valid


def run_match(f, fwd, peak, valid, cond, rng, label):
    tod = f["tod"].to_numpy(); month = f["month"].to_numpy()
    day = f["srv_date"].to_numpy()
    out = {}
    for oname, ovals in (("spike", (peak >= SPIKE_ABN).astype(float)),
                         ("fwd_abn", np.nan_to_num(fwd))):
        r = S.matched_baseline(ovals, valid, cond & valid, tod, month, day, rng=rng)
        if r:
            out[oname] = r
    out["label"] = label
    return out


def main():
    rng = np.random.default_rng(41)
    report = {}

    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        fwd, peak, valid = forward_measures(f)
        n = len(f)
        res = {}

        # ---------- Part 6: raw vs clock-standardised compression --------
        head = {}
        for tag in ("raw", "adj"):
            cond = (f[f"state_{tag}"] == "compressed").to_numpy()
            head[tag] = run_match(f, fwd, peak, valid, cond, rng, f"{tag}_compressed")
        # continuous version: quintiles of abn (volatility / normal for this slot)
        abn = f["abn"].to_numpy()
        qs = np.nanpercentile(abn[valid], [20, 40, 60, 80])
        for i in range(5):
            lo = -np.inf if i == 0 else qs[i - 1]
            hi = np.inf if i == 4 else qs[i]
            cond = (abn >= lo) & (abn < hi)
            head[f"abn_q{i+1}"] = run_match(f, fwd, peak, valid, cond, rng,
                                            f"abn quintile {i+1}")
        res["part6_head_to_head"] = head

        # ---------- Parts 3 & 4: depth and duration ----------------------
        ep = pd.read_pickle(OUT / f"epi_{sym}_adj.pkl")
        idx = ep["end_idx"].to_numpy()
        depth = {}
        for name, lo, hi in (("d0-5", 0, 5), ("d5-10", 5, 10),
                             ("d10-15", 10, 15), ("d15-20", 15, 20)):
            sel = idx[(ep["min_pct"] >= lo) & (ep["min_pct"] < hi)]
            c = np.zeros(n, bool); c[sel] = True
            depth[name] = run_match(f, fwd, peak, valid, c, rng, name)
        res["part3_depth"] = depth

        dur = {}
        for name, lo, hi in (("1-3", 1, 4), ("4-6", 4, 7), ("7-12", 7, 13),
                             ("13-24", 13, 25), ("25+", 25, 10**6)):
            sel = idx[(ep["duration"] >= lo) & (ep["duration"] < hi)]
            c = np.zeros(n, bool); c[sel] = True
            dur[name] = run_match(f, fwd, peak, valid, c, rng, name)
        res["part4_duration"] = dur

        report[sym] = res

        print(f"\n=== {sym} ===")
        print("  PART 6 -- which compression variable carries information?")
        for k in ("raw", "adj"):
            s = head[k].get("spike"); a = head[k].get("fwd_abn")
            if s:
                print(f"    {k:8s} n={s['n']:>6}  P(spike)={s['observed']:.3f} vs matched "
                      f"{s['matched']:.3f}  lift={s['lift']:.3f} "
                      f"CI[{s['lift_ci_lo']:.3f},{s['lift_ci_hi']:.3f}] p={s['p_two_sided']:.3f}"
                      f"   fwd_abn lift={a['lift']:.3f} CI[{a['lift_ci_lo']:.3f},{a['lift_ci_hi']:.3f}]")
        print("    abn quintiles (q1 = quietest for this clock time):")
        for i in range(5):
            s = head[f"abn_q{i+1}"].get("spike"); a = head[f"abn_q{i+1}"].get("fwd_abn")
            if s:
                print(f"      q{i+1} n={s['n']:>6} P(spike)={s['observed']:.3f} matched={s['matched']:.3f} "
                      f"lift={s['lift']:.3f} CI[{s['lift_ci_lo']:.3f},{s['lift_ci_hi']:.3f}]"
                      f"   fwd_abn lift={a['lift']:.3f}")
        print("  PART 3 -- depth (matched):")
        for k, v in depth.items():
            s = v.get("spike")
            if s:
                print(f"    {k:8s} n={s['n']:>5} lift={s['lift']:.3f} "
                      f"CI[{s['lift_ci_lo']:.3f},{s['lift_ci_hi']:.3f}] p={s['p_two_sided']:.3f}")
        print("  PART 4 -- duration already compressed (matched):")
        for k in ("1-3", "4-6", "7-12", "13-24", "25+"):
            s = dur[k].get("spike"); a = dur[k].get("fwd_abn")
            if s:
                print(f"    {k:8s} n={s['n']:>5} spike lift={s['lift']:.3f} "
                      f"CI[{s['lift_ci_lo']:.3f},{s['lift_ci_hi']:.3f}] p={s['p_two_sided']:.3f}"
                      f"   fwd_abn lift={a['lift']:.3f} CI[{a['lift_ci_lo']:.3f},{a['lift_ci_hi']:.3f}]")
                RL.register(
                    test_id=f"P4.dur.{sym}.{k}", part="4",
                    hypothesis="Longer-persisting compression raises the odds of a clock-referenced volatility spike",
                    definition=f"P(max bar range / seasonal range >= 2 within 24 bars) after a compression episode of {k} bars ends",
                    instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                    n=s["n"], p_value=s["p_two_sided"],
                    ci=(s["lift_ci_lo"], s["lift_ci_hi"]),
                    control="5-min slot x calendar month matched baseline, day-block bootstrap",
                    result={"lift": s["lift"], "observed": s["observed"],
                            "matched": s["matched"]})
        for tag in ("raw", "adj"):
            s = head[tag].get("spike")
            if s:
                RL.register(
                    test_id=f"P6.headtohead.{sym}.{tag}", part="6",
                    hypothesis="Abnormal (clock-standardised) compression predicts volatility spikes better than raw low ATR",
                    definition=f"P(clock-referenced spike within 24 bars) | {tag} compression",
                    instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                    n=s["n"], p_value=s["p_two_sided"],
                    ci=(s["lift_ci_lo"], s["lift_ci_hi"]),
                    control="5-min slot x calendar month matched baseline",
                    result={"lift": s["lift"]})

    with (OUT / "r3_conditional.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r3_conditional.json")


if __name__ == "__main__":
    main()
