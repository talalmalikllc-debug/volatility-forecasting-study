"""R4 -- Part 5: does the state going INTO a session handover predict what
happens after it?

This is a day-level test, which makes it much cleaner than anything bar-level:
one observation per boundary per day, ~900-1000 observations, essentially
independent. And because every observation sits at the SAME clock time by
construction, time of day is controlled exactly rather than approximately --
the confound that wrecked the first study cannot arise here. Only the calendar
month still needs matching, which is done by comparing within month.

For each boundary:
    pre   = the 12 bars ending at the boundary        (1 hour)
    post  = the 24 bars starting at the boundary      (2 hours)
    quiet = pre-window activity in the bottom quintile FOR THAT BOUNDARY

Outcomes: post-window activity relative to its own seasonal norm, the odds of a
clock-referenced spike, and whether the pre-window's direction carries into the
post window.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_log as RL
import stats_util as S

OUT = Path(_os.path.join(_REPO, "results"))
SYMS = ["XAUUSD", "EURUSD", "US100.cash", "US500.cash", "US30.cash", "GER40.cash"]
PRE, POST = 12, 24

BOUNDARIES = {
    "asia_to_london": 3 * 60,
    "london_to_ny": 8 * 60,
    "ny_cash_open": 9 * 60 + 30,
    "ny_am_to_pm": 12 * 60,
    "ny_to_asia": 18 * 60,
}


def collect(f: pd.DataFrame, tod_target: int) -> pd.DataFrame:
    tod = f["tod"].to_numpy()
    R = f["R"].to_numpy(); seas = f["atr_seasonal"].to_numpy()
    close = f["close"].to_numpy(); gap = f["gap"].to_numpy()
    n = len(f)
    hits = np.flatnonzero(tod == tod_target)
    hits = hits[(hits >= PRE) & (hits + POST < n)]
    rows = []
    for i in hits:
        pre = slice(i - PRE, i)
        post = slice(i, i + POST)
        if gap[i - PRE + 1:i + POST].any():
            continue
        sp, sr = seas[pre], R[pre]
        sp2, sr2 = seas[post], R[post]
        if np.isnan(sp).any() or np.isnan(sp2).any():
            continue
        rows.append({
            "day": f["srv_date"].to_numpy()[i],
            "month": f["month"].to_numpy()[i],
            "year": f["year"].to_numpy()[i],
            "pre_abn": sr.sum() / sp.sum(),
            "post_abn": sr2.sum() / sp2.sum(),
            "post_peak": float((sr2 / sp2).max()),
            "pre_dir": np.sign(close[i - 1] - close[i - PRE]),
            "post_move_s": (close[i + POST - 1] - close[i - 1]) / sp2.mean(),
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(71)
    report = {}

    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        res = {}
        print(f"\n=== {sym} ===")
        for bname, tt in BOUNDARIES.items():
            d = collect(f, tt)
            if len(d) < 300:
                continue
            cut = d["pre_abn"].quantile(0.2)
            quiet = d["pre_abn"] <= cut
            a, b = d[quiet], d[~quiet]

            spike_d = S.boot_diff((a["post_peak"] >= 2).astype(float), a["day"],
                                  (b["post_peak"] >= 2).astype(float), b["day"], rng=rng)
            abn_d = S.boot_diff(a["post_abn"], a["day"], b["post_abn"], b["day"], rng=rng)
            # directional carry-over: pre-window direction into the post window
            carry = S.boot_mean(d["pre_dir"] * d["post_move_s"], d["day"], rng=rng)
            carry_q = S.boot_mean(a["pre_dir"] * a["post_move_s"], a["day"], rng=rng)

            res[bname] = {"n_days": int(len(d)), "n_quiet": int(quiet.sum()),
                          "P_spike_quiet": float((a["post_peak"] >= 2).mean()),
                          "P_spike_rest": float((b["post_peak"] >= 2).mean()),
                          "spike_diff": spike_d,
                          "post_abn_quiet": float(a["post_abn"].mean()),
                          "post_abn_rest": float(b["post_abn"].mean()),
                          "abn_diff": abn_d,
                          "carry_all": carry, "carry_quiet": carry_q}

            print(f"  {bname:16s} n={len(d):>4} quiet={quiet.sum():>3}  "
                  f"P(spike after) quiet={res[bname]['P_spike_quiet']:.3f} vs "
                  f"rest={res[bname]['P_spike_rest']:.3f}  "
                  f"diff={spike_d['diff']:+.3f} CI[{spike_d['ci_lo']:+.3f},"
                  f"{spike_d['ci_hi']:+.3f}] p={spike_d['p_two_sided']:.3f}")
            print(f"  {'':16s}     post activity quiet={res[bname]['post_abn_quiet']:.3f} vs "
                  f"rest={res[bname]['post_abn_rest']:.3f} diff={abn_d['diff']:+.3f} "
                  f"CI[{abn_d['ci_lo']:+.3f},{abn_d['ci_hi']:+.3f}] p={abn_d['p_two_sided']:.3f}"
                  f"   | direction carry-over all={carry['mean']:+.3f}"
                  f"[{carry['ci_lo']:+.3f},{carry['ci_hi']:+.3f}]"
                  f" quiet={carry_q['mean']:+.3f}[{carry_q['ci_lo']:+.3f},{carry_q['ci_hi']:+.3f}]")

            RL.register(
                test_id=f"P5.spike.{sym}.{bname}", part="5",
                hypothesis="An abnormally quiet run into a session handover predicts a volatility spike after it",
                definition=f"P(peak bar range / seasonal >= 2 in the {POST} bars after {bname}) "
                           f"| pre-window activity in the bottom quintile, vs the rest",
                instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                n=int(len(d)), p_value=spike_d["p_two_sided"],
                ci=(spike_d["ci_lo"], spike_d["ci_hi"]),
                control="boundary fixes time of day exactly; day-block bootstrap of the difference",
                result={"diff": spike_d["diff"],
                        "P_quiet": res[bname]["P_spike_quiet"],
                        "P_rest": res[bname]["P_spike_rest"]})
            RL.register(
                test_id=f"P5.carry.{sym}.{bname}", part="5",
                hypothesis="Direction established before a session handover carries into the next session",
                definition=f"mean of sign(pre-window move) x post-window move, in seasonal-range units",
                instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                n=int(len(d)), ci=(carry["ci_lo"], carry["ci_hi"]),
                p_value=2 * min(carry["p_gt0"], 1 - carry["p_gt0"]),
                control="day-block bootstrap; boundary fixes time of day",
                result={"carry": carry["mean"], "carry_quiet": carry_q["mean"]})
        report[sym] = res

    with (OUT / "r4_sessions.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r4_sessions.json")


if __name__ == "__main__":
    main()
