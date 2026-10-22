"""R6 -- Parts 11, 12, 14 applied to the surviving phenomenon.

Part 5 produced a large, clean effect: an abnormally quiet hour into a session
handover forecasts an abnormally quiet session after it. Three things decide
whether that is worth anything.

1. TEMPORAL STABILITY (Part 12). Split by year. The first study was nearly
   derailed by a pooled statistic that no individual year supported.

2. A CAUSAL FLAG. The Part 5 test used a quintile threshold computed over the
   whole sample, which a live trader could not know. Here the threshold is a
   trailing 250-session quantile, shifted, so the flag is one a trader could
   actually have set that morning. If the effect only exists with hindsight
   thresholds it is not an effect.

3. MONETISABILITY (Part 14). A volatility forecast is not a return. The natural
   consequence of "this session will be quiet" is that the session's opening
   range should contain price better than usual, so the probe is: after price
   first leaves the opening range, does it revert to the range midpoint before
   extending a full range-width further? That is a pure path statistic with no
   free parameters to tune, and it converts directly into an expectancy once
   the spread is charged.
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
PRE, OR_BARS, POST = 12, 12, 36
BOUNDARIES = {"asia_to_london": 3 * 60, "ny_cash_open": 9 * 60 + 30}
TRAIL = 250          # sessions in the trailing threshold
Q = 0.20


def collect(f: pd.DataFrame, tt: int) -> pd.DataFrame:
    tod = f["tod"].to_numpy(); R = f["R"].to_numpy(); seas = f["atr_seasonal"].to_numpy()
    high = f["high"].to_numpy(); low = f["low"].to_numpy(); close = f["close"].to_numpy()
    gap = f["gap"].to_numpy(); spread = f["spread_px"].to_numpy()
    n = len(f)
    hits = np.flatnonzero(tod == tt)
    hits = hits[(hits >= PRE) & (hits + OR_BARS + POST < n)]
    rows = []
    for i in hits:
        if gap[i - PRE + 1:i + OR_BARS + POST].any():
            continue
        sp, sr = seas[i - PRE:i], R[i - PRE:i]
        s2, r2 = seas[i:i + OR_BARS + POST], R[i:i + OR_BARS + POST]
        if np.isnan(sp).any() or np.isnan(s2).any():
            continue
        oh = high[i:i + OR_BARS].max()
        ol = low[i:i + OR_BARS].min()
        if oh <= ol:
            continue
        rows.append({
            "i": i, "day": f["srv_date"].to_numpy()[i],
            "year": f["year"].to_numpy()[i], "month": f["month"].to_numpy()[i],
            "pre_abn": sr.sum() / sp.sum(),
            "post_abn": r2.sum() / s2.sum(),
            "post_peak": float((r2 / s2).max()),
            "or_hi": oh, "or_lo": ol, "or_size": oh - ol,
            "spread": float(spread[i:i + OR_BARS + POST].mean()),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    # CAUSAL flag: trailing quantile of pre_abn, shifted so today is excluded
    d["thr"] = d["pre_abn"].shift(1).rolling(TRAIL, min_periods=120).quantile(Q)
    d["quiet"] = d["pre_abn"] <= d["thr"]
    return d.dropna(subset=["thr"])


def containment(f: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """After price first leaves the opening range, does it revert to the range
    midpoint before extending a further full range-width?"""
    high = f["high"].to_numpy(); low = f["low"].to_numpy()
    out = []
    for _, r in d.iterrows():
        i = int(r["i"]) + OR_BARS
        hh = high[i:i + POST]; ll = low[i:i + POST]
        oh, ol, sz = r["or_hi"], r["or_lo"], r["or_size"]
        mid = (oh + ol) / 2
        up = np.flatnonzero(hh > oh)
        dn = np.flatnonzero(ll < ol)
        fu = up[0] if len(up) else 10**6
        fd = dn[0] if len(dn) else 10**6
        if fu == fd == 10**6:
            out.append({"left": False, "revert": np.nan, "net_R": np.nan})
            continue
        if fu < fd:
            side, t0, ext, tgt = 1, fu, oh + sz, mid
            rev = np.flatnonzero(ll[t0:] <= tgt)
            ex = np.flatnonzero(hh[t0:] >= ext)
        else:
            side, t0, ext, tgt = -1, fd, ol - sz, mid
            rev = np.flatnonzero(hh[t0:] >= tgt)
            ex = np.flatnonzero(ll[t0:] <= ext)
        tr = rev[0] if len(rev) else 10**6
        te = ex[0] if len(ex) else 10**6
        # fading the break: win = midpoint first, loss = a further range-width
        if tr < te:
            gross = (sz / 2)
        elif te < tr:
            gross = -sz
        else:
            gross = 0.0
        out.append({"left": True, "revert": float(tr < te),
                    "net_R": (gross - r["spread"]) / sz})
    return pd.DataFrame(out, index=d.index)


def main():
    rng = np.random.default_rng(83)
    report = {}

    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        res = {}
        print(f"\n=== {sym} ===")
        for bname, tt in BOUNDARIES.items():
            d = collect(f, tt)
            if d.empty or d["quiet"].sum() < 100:
                continue
            c = containment(f, d)
            d = pd.concat([d, c], axis=1)
            a, b = d[d.quiet], d[~d.quiet]

            spike = S.boot_diff((a["post_peak"] >= 2).astype(float), a["day"],
                                (b["post_peak"] >= 2).astype(float), b["day"], rng=rng)
            left = S.boot_diff(a["left"].astype(float), a["day"],
                               b["left"].astype(float), b["day"], rng=rng)
            av, bv = a.dropna(subset=["revert"]), b.dropna(subset=["revert"])
            rev = S.boot_diff(av["revert"], av["day"], bv["revert"], bv["day"], rng=rng)
            net_q = S.boot_mean(av["net_R"], av["day"], rng=rng)
            net_r = S.boot_mean(bv["net_R"], bv["day"], rng=rng)

            per_year = {}
            for y, g in d.groupby("year", observed=True):
                gq, gr = g[g.quiet], g[~g.quiet]
                if len(gq) < 20:
                    continue
                per_year[int(y)] = {
                    "n_quiet": int(len(gq)),
                    "P_spike_quiet": float((gq["post_peak"] >= 2).mean()),
                    "P_spike_rest": float((gr["post_peak"] >= 2).mean()),
                    "post_abn_quiet": float(gq["post_abn"].mean()),
                    "post_abn_rest": float(gr["post_abn"].mean()),
                }

            res[bname] = {"n": int(len(d)), "n_quiet": int(d["quiet"].sum()),
                          "spike_diff": spike, "left_range_diff": left,
                          "revert_diff": rev, "fade_net_quiet": net_q,
                          "fade_net_rest": net_r, "per_year": per_year,
                          "mean_or_size_over_spread_quiet":
                              float((a["or_size"] / a["spread"]).mean()),
                          "mean_or_size_over_spread_rest":
                              float((b["or_size"] / b["spread"]).mean())}

            print(f"  {bname} (causal trailing-{TRAIL} flag): n={len(d)} quiet={d['quiet'].sum()}")
            print(f"    P(spike) diff = {spike['diff']:+.3f} "
                  f"CI[{spike['ci_lo']:+.3f},{spike['ci_hi']:+.3f}] p={spike['p_two_sided']:.3f}")
            print(f"    per year P(spike) quiet vs rest: " + "  ".join(
                f"{y}:{v['P_spike_quiet']:.2f}/{v['P_spike_rest']:.2f}"
                for y, v in per_year.items()))
            print(f"    P(leaves opening range) diff = {left['diff']:+.3f} "
                  f"CI[{left['ci_lo']:+.3f},{left['ci_hi']:+.3f}] p={left['p_two_sided']:.3f}")
            print(f"    P(reverts to mid before extending) quiet={av['revert'].mean():.3f} "
                  f"rest={bv['revert'].mean():.3f} diff={rev['diff']:+.3f} "
                  f"CI[{rev['ci_lo']:+.3f},{rev['ci_hi']:+.3f}] p={rev['p_two_sided']:.3f}")
            print(f"    fade expectancy NET of spread: quiet={net_q['mean']:+.4f} "
                  f"CI[{net_q['ci_lo']:+.4f},{net_q['ci_hi']:+.4f}]  "
                  f"rest={net_r['mean']:+.4f} CI[{net_r['ci_lo']:+.4f},{net_r['ci_hi']:+.4f}]"
                  f"   (opening range = {res[bname]['mean_or_size_over_spread_quiet']:.0f}x spread when quiet)")

            RL.register(
                test_id=f"P12.spike.{sym}.{bname}", part="11/12",
                hypothesis="The quiet-session forecast survives a causal threshold and is stable across years",
                definition=f"P(clock-referenced spike after {bname}) | pre-window in the bottom "
                           f"quintile of a TRAILING {TRAIL}-session distribution, vs the rest",
                instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                n=int(len(d)), p_value=spike["p_two_sided"],
                ci=(spike["ci_lo"], spike["ci_hi"]),
                control="boundary fixes time of day; trailing causal threshold; per-year split reported",
                result={"diff": spike["diff"], "per_year": per_year})
            RL.register(
                test_id=f"P14.fade.{sym}.{bname}", part="14",
                hypothesis="The quiet forecast monetises as better opening-range containment",
                definition="expectancy of fading the first opening-range break to the range midpoint, "
                           "stop one range-width beyond, net of spread, in range units",
                instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                n=net_q["n"], ci=(net_q["ci_lo"], net_q["ci_hi"]),
                p_value=2 * min(net_q["p_gt0"], 1 - net_q["p_gt0"]),
                control="same probe on non-flagged sessions",
                result={"net_quiet": net_q["mean"], "net_rest": net_r["mean"]})
        report[sym] = res

    with (OUT / "r6_consequence.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r6_consequence.json")


if __name__ == "__main__":
    main()
