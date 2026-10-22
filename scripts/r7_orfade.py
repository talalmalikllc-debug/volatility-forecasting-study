"""R7 -- focused, honest test of the one lead that emerged.

FLAGGED AS EXPLORATORY. This was not in the research plan. It surfaced while
looking at the control arm of the R6 monetisability probe, where fading the
first break of the opening range showed positive net expectancy on five of six
instruments. Because it was found by looking at the data, the p-values below
carry no confirmatory weight; the purpose here is to establish whether it is
solid enough to deserve a pre-registered test on data this study has not seen.

Two things are fixed relative to R6:

  * UNRESOLVED TRADES ARE MARKED TO MARKET. R6 scored a trade that reached
    neither the midpoint nor the extension as exactly zero. That is not a
    result, it is an accounting convention, and since it covered ~49% of cases
    it dominated the expectancy. Here the position is closed at the last bar of
    the window at whatever price is there.
  * Both a 1-hour and a 3-hour opening range are run, and the entry level is
    tested with and without a buffer, to check the effect is not living on one
    parameter choice.
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
BOUNDARY = {"ny_cash_open": 9 * 60 + 30, "asia_to_london": 3 * 60}
POST = 36


def probe(f, tt, or_bars, buffer_frac):
    tod = f["tod"].to_numpy()
    high = f["high"].to_numpy(); low = f["low"].to_numpy(); close = f["close"].to_numpy()
    gap = f["gap"].to_numpy(); spread = f["spread_px"].to_numpy()
    n = len(f)
    hits = np.flatnonzero(tod == tt)
    hits = hits[(hits >= 2) & (hits + or_bars + POST < n)]
    rows = []
    for i in hits:
        if gap[i + 1:i + or_bars + POST].any():
            continue
        oh = high[i:i + or_bars].max(); ol = low[i:i + or_bars].min()
        sz = oh - ol
        if sz <= 0:
            continue
        j = i + or_bars
        hh = high[j:j + POST]; ll = low[j:j + POST]; cc = close[j:j + POST]
        up_lvl = oh + buffer_frac * sz
        dn_lvl = ol - buffer_frac * sz
        u = np.flatnonzero(hh > up_lvl); d = np.flatnonzero(ll < dn_lvl)
        fu = u[0] if len(u) else 10**6
        fd = d[0] if len(d) else 10**6
        if fu == fd == 10**6:
            continue
        mid = (oh + ol) / 2
        sp = float(spread[j:j + POST].mean())
        if fu < fd:
            side, t0, entry = -1, fu, up_lvl          # fade the upside break
            ext = up_lvl + sz
            rev = np.flatnonzero(ll[t0:] <= mid)
            ex = np.flatnonzero(hh[t0:] >= ext)
        else:
            side, t0, entry = +1, fd, dn_lvl
            ext = dn_lvl - sz
            rev = np.flatnonzero(hh[t0:] >= mid)
            ex = np.flatnonzero(ll[t0:] <= ext)
        tr = rev[0] if len(rev) else 10**6
        te = ex[0] if len(ex) else 10**6
        if tr < te:
            gross = abs(entry - mid)
            outcome = "target"
        elif te < tr:
            gross = -sz
            outcome = "stop"
        else:
            # marked to market at the end of the window -- not booked at zero
            gross = side * (cc[-1] - entry)
            outcome = "timeout"
        rows.append({"day": f["srv_date"].to_numpy()[i],
                     "year": f["year"].to_numpy()[i],
                     "net_R": (gross - sp) / sz, "gross_R": gross / sz,
                     "cost_R": sp / sz, "outcome": outcome, "side": side})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(97)
    report = {}
    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        res = {}
        print(f"\n=== {sym} ===")
        for bname, tt in BOUNDARY.items():
            for or_bars in (12, 36):
                for buf in (0.0, 0.1):
                    d = probe(f, tt, or_bars, buf)
                    if len(d) < 300:
                        continue
                    b = S.boot_mean(d["net_R"], d["day"], rng=rng)
                    key = f"{bname}_or{or_bars}_buf{buf}"
                    py = {int(y): float(g["net_R"].mean())
                          for y, g in d.groupby("year", observed=True) if len(g) > 40}
                    res[key] = {"boot": b, "per_year": py,
                                "P_target": float((d.outcome == "target").mean()),
                                "P_stop": float((d.outcome == "stop").mean()),
                                "P_timeout": float((d.outcome == "timeout").mean()),
                                "cost_R": float(d["cost_R"].mean())}
                    ok = "  <<<" if b["ci_lo"] > 0 else ""
                    print(f"  {key:34s} n={b['n']:>4} net={b['mean']:+.4f}R "
                          f"CI[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}] cost={res[key]['cost_R']:.4f}R "
                          f"| tgt/stop/timeout={res[key]['P_target']:.2f}/"
                          f"{res[key]['P_stop']:.2f}/{res[key]['P_timeout']:.2f}{ok}")
                    print(f"  {'':34s} per year: " +
                          "  ".join(f"{y}:{v:+.3f}" for y, v in py.items()))
                    RL.register(
                        test_id=f"EXPL.orfade.{sym}.{key}", part="exploratory",
                        hypothesis="Fading the first break of the opening range has positive net expectancy",
                        definition=f"fade first break of the {or_bars}-bar range after {bname} "
                                   f"(buffer {buf} range), target midpoint, stop one range-width, "
                                   f"unresolved marked to market at bar {POST}",
                        instrument=sym, sample="M5 2022-01..2026-07",
                        prespecified=False, n=b["n"],
                        ci=(b["ci_lo"], b["ci_hi"]),
                        p_value=2 * min(b["p_gt0"], 1 - b["p_gt0"]),
                        control="none -- discovered post hoc, reported for follow-up only",
                        result={"net_R": b["mean"], "per_year": py})
        report[sym] = res
    with (OUT / "r7_orfade.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r7_orfade.json")


if __name__ == "__main__":
    main()
