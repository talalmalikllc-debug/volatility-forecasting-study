"""R2 -- Parts 2, 3, 4: episode-level transition taxonomy, depth, duration.

An "episode" is a maximal run of compressed bars. We locate the bar where the
episode ENDS and then describe, in detail, what the next two hours look like.

The yardstick matters. Study 1 measured forward expansion against the symbol's
global median ATR, which conflates "the market woke up" with "the clock moved
into a busier part of the day". Here every forward quantity is divided by the
SEASONAL expectation for the exact five-minute slots it spans:

    fwd_abn = sum(range over the window) / sum(seasonal range for those slots)

fwd_abn = 1.0 means "exactly as busy as that stretch of the clock normally is".
That is the only definition under which "expansion" is a claim about the market
rather than a claim about what time it is.

Taxonomy is reported as a 3 x 5 contingency (volatility outcome x directional
outcome) and then mapped onto the ten categories requested.
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
W = 24                    # post-episode window, 2 hours of M5
SPIKE = 2.5               # bar range / ATR at episode end that counts as a spike
DEPTH_EDGES = [0, 5, 10, 15, 20]
DUR_BUCKETS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 10**6)]


def build_episodes(f: pd.DataFrame, tag: str) -> pd.DataFrame:
    st = f[f"state_{tag}"].to_numpy()
    eid = f[f"epi_{tag}"].to_numpy()
    pct = f[f"pct_{'raw' if tag == 'raw' else 'adj'}"].to_numpy()
    close = f["close"].to_numpy()
    high = f["high"].to_numpy()
    low = f["low"].to_numpy()
    R = f["R"].to_numpy()
    seas = f["atr_seasonal"].to_numpy()
    atr = f["atr"].to_numpy()
    gap = f["gap"].to_numpy()
    n = len(f)

    comp = st == "compressed"
    # episode end = last bar of each run
    end = comp & ~np.roll(comp, -1)
    end[-1] = False
    ends = np.flatnonzero(end)
    ends = ends[(ends + W < n - 1) & (ends > 300)]
    if len(ends) == 0:
        return pd.DataFrame()

    contig = V.contiguous_forward(gap, W)
    ends = ends[contig[ends]]
    # the seasonal profile needs a warm-up and can be missing in thin slots;
    # a single NaN inside the window would silently poison every forward mean
    seas_ok = ~np.isnan(sliding_window_view(seas, W)[ends + 1]).any(axis=1)
    ends = ends[seas_ok & ~np.isnan(seas[ends])]
    if len(ends) == 0:
        return pd.DataFrame()

    # ---- window views: bars e+1 .. e+W ---------------------------------
    hw = sliding_window_view(high, W)[ends + 1]
    lw = sliding_window_view(low, W)[ends + 1]
    cw = sliding_window_view(close, W)[ends + 1]
    rw = sliding_window_view(R, W)[ends + 1]
    sw = sliding_window_view(seas, W)[ends + 1]
    pw = sliding_window_view(pct, W)[ends + 1]

    atr_e = atr[ends]
    c_e = close[ends]

    # episode geometry
    starts = np.empty_like(ends)
    dur = np.empty_like(ends)
    minpct = np.empty(len(ends))
    box_hi = np.empty(len(ends))
    box_lo = np.empty(len(ends))
    for i, e in enumerate(ends):
        this = eid[e]
        s = e
        while s > 0 and eid[s - 1] == this:
            s -= 1
        starts[i] = s
        dur[i] = e - s + 1
        minpct[i] = np.nanmin(pct[s:e + 1])
        box_hi[i] = high[s:e + 1].max()
        box_lo[i] = low[s:e + 1].min()

    # ---- volatility outcome --------------------------------------------
    fwd_abn = rw.sum(1) / sw.sum(1)                 # busier than normal for the clock?
    per_bar_abn = rw / sw
    peak_abn = per_bar_abn.max(1)
    t_peak_abn = per_bar_abn.argmax(1) + 1
    peak_pct = np.nanmax(pw, axis=1)
    spike_ratio = rw.max(1) / atr_e
    t_spike = rw.argmax(1) + 1

    # `spike_ratio` divides by the ATR at episode end, which is SMALLER the
    # deeper and longer the compression has been. Bucketing by depth or duration
    # and then reading P(spike_ratio > 2.5) therefore measures the denominator,
    # not the market. `peak_abn` clears a fixed, clock-referenced bar instead and
    # is the honest version of the same question.
    vol_out = np.where(peak_abn >= 2.0, "spike",
                       np.where(peak_pct >= 40, "gradual", "stays_low"))
    vol_out_selfref = np.where(spike_ratio >= SPIKE, "spike",
                               np.where(peak_pct >= 40, "gradual", "stays_low"))

    # ---- directional outcome -------------------------------------------
    up = cw > box_hi[:, None]
    dn = cw < box_lo[:, None]
    big = W + 1
    f_up = np.where(up.any(1), up.argmax(1), big)
    f_dn = np.where(dn.any(1), dn.argmax(1), big)
    exited = (f_up < big) | (f_dn < big)
    dirn = np.where(f_up < f_dn, 1, np.where(f_dn < f_up, -1, 0))
    t_exit = np.minimum(f_up, f_dn)

    # from the exit bar to the end of the window, signed by breakout direction
    idx = np.clip(t_exit, 0, W - 1)
    c_exit = cw[np.arange(len(ends)), idx]
    c_end = cw[:, -1]
    post = np.where(exited, dirn * (c_end - c_exit) / atr_e, np.nan)

    # MFE / MAE after the exit, signed by breakout direction
    sgn = dirn[:, None]
    fav = np.where(sgn > 0, hw, -lw)
    adv = np.where(sgn > 0, lw, -hw)
    mask = np.arange(W)[None, :] >= idx[:, None]
    fav_m = np.where(mask, fav, -np.inf)
    adv_m = np.where(mask, adv, np.inf)
    e_s = dirn * c_exit
    mfe = np.where(exited, (fav_m.max(1) - e_s) / atr_e, np.nan)
    mae = np.where(exited, (e_s - adv_m.min(1)) / atr_e, np.nan)

    dir_out = np.where(~exited, "no_exit",
                       np.where(dirn > 0,
                                np.where(post > 0, "up_cont", "up_rev"),
                                np.where(post > 0, "down_cont", "down_rev")))

    return pd.DataFrame({
        "end_idx": ends, "start_idx": starts, "duration": dur,
        "min_pct": minpct, "atr_e": atr_e, "abn_e": f["abn"].to_numpy()[ends],
        "srv_date": f["srv_date"].to_numpy()[ends],
        "month": f["month"].to_numpy()[ends], "year": f["year"].to_numpy()[ends],
        "tod": f["tod"].to_numpy()[ends], "session": f["session"].to_numpy()[ends],
        "fwd_abn": fwd_abn, "peak_abn": peak_abn, "t_peak_abn": t_peak_abn,
        "peak_pct": peak_pct, "spike_ratio": spike_ratio, "t_spike": t_spike,
        "vol_out": vol_out, "vol_out_selfref": vol_out_selfref,
        "dir_out": dir_out, "exited": exited,
        "dirn": dirn, "t_exit": np.where(exited, t_exit + 1, np.nan),
        "post_ret_atr": post, "mfe_atr": mfe, "mae_atr": mae,
        "box_h_atr": (box_hi - box_lo) / atr_e,
        "term_ret_atr": (c_end - c_e) / atr_e,
    })


def all_bar_windows(f: pd.DataFrame):
    """The same forward measurements for EVERY bar, to serve as the baseline.

    Without this the episode numbers float free: fwd_abn is a ratio of a range
    to a MEDIAN range, and range distributions are right-skewed, so its
    unconditional mean sits above 1.0 for reasons that have nothing to do with
    compression. The baseline has to be measured, not assumed.
    """
    R = f["R"].to_numpy()
    seas = f["atr_seasonal"].to_numpy()
    gap = f["gap"].to_numpy()
    n = len(f)
    rw = sliding_window_view(R, W)
    sw = sliding_window_view(seas, W)
    fwd = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    m = n - W - 1
    ok = ~np.isnan(sw[1:m + 1]).any(axis=1)
    fwd[:m] = np.where(ok, rw[1:m + 1].sum(1) / sw[1:m + 1].sum(1), np.nan)
    peak[:m] = np.where(ok, (rw[1:m + 1] / sw[1:m + 1]).max(1), np.nan)
    valid[:m] = ok & V.contiguous_forward(gap, W)[:m]
    return fwd, peak, valid


def bucket_stats(ep: pd.DataFrame, by: str, rng) -> dict:
    out = {}
    for name, g in ep.groupby(by, observed=True):
        if len(g) < 100:
            continue
        d = g["srv_date"].to_numpy()
        out[str(name)] = {
            "n": int(len(g)),
            "fwd_abn": S.boot_mean(g["fwd_abn"], d, rng=rng),
            "peak_abn": float(g["peak_abn"].mean()),
            "P_spike": float((g["vol_out"] == "spike").mean()),
            "P_spike_selfref": float((g["vol_out_selfref"] == "spike").mean()),
            "P_stays_low": float((g["vol_out"] == "stays_low").mean()),
            "P_exit": float(g["exited"].mean()),
            "P_cont_given_exit": float(g.loc[g.exited, "dir_out"]
                                       .isin(["up_cont", "down_cont"]).mean()),
            "post_ret_atr": S.boot_mean(g.loc[g.exited, "post_ret_atr"],
                                        g.loc[g.exited, "srv_date"], rng=rng),
            "mfe_atr": float(g["mfe_atr"].mean(skipna=True)),
            "mae_atr": float(g["mae_atr"].mean(skipna=True)),
            "t_exit": float(g["t_exit"].mean(skipna=True)),
            "t_peak_abn": float(g["t_peak_abn"].mean()),
            "median_duration": float(g["duration"].median()),
        }
    return out


def main():
    rng = np.random.default_rng(23)
    report = {}

    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        bfwd, bpeak, bvalid = all_bar_windows(f)
        base = {
            "fwd_abn_uncond": float(np.nanmean(bfwd[bvalid])),
            "P_spike_uncond": float(np.nanmean((bpeak[bvalid] >= 2.0))),
        }
        res = {"baseline_all_bars": base}
        for tag in ("adj", "raw"):
            ep = build_episodes(f, tag)
            if ep.empty:
                continue
            ep.to_pickle(OUT / f"epi_{sym}_{tag}.pkl")
            d = ep["srv_date"].to_numpy()

            ct = pd.crosstab(ep["vol_out"], ep["dir_out"], normalize=True)
            depth_b = pd.cut(ep["min_pct"], DEPTH_EDGES, right=False)
            dur_b = pd.cut(ep["duration"],
                           [1, 4, 7, 13, 25, 10**6], right=False,
                           labels=["1-3", "4-6", "7-12", "13-24", "25+"])
            depth_b = pd.cut(ep["min_pct"], DEPTH_EDGES, right=False,
                             labels=["d0-5", "d5-10", "d10-15", "d15-20"])
            ep = ep.assign(depth_b=depth_b.astype(str), dur_b=dur_b.astype(str))

            # matched (5-min slot x month) comparison of the spike probability
            idx = ep["end_idx"].to_numpy()
            cond = np.zeros(len(f), dtype=bool)
            cond[idx] = True
            mb = S.matched_baseline(bpeak >= 2.0, bvalid, cond & bvalid,
                                    f["tod"].to_numpy(), f["month"].to_numpy(),
                                    f["srv_date"].to_numpy(), rng=rng)

            res[tag] = {
                "n_episodes": int(len(ep)),
                "spike_vs_matched": mb,
                "overall": {
                    "fwd_abn": S.boot_mean(ep["fwd_abn"], d, rng=rng),
                    "P_spike": float((ep["vol_out"] == "spike").mean()),
                    "P_gradual": float((ep["vol_out"] == "gradual").mean()),
                    "P_stays_low": float((ep["vol_out"] == "stays_low").mean()),
                    "P_exit": float(ep["exited"].mean()),
                    "P_cont_given_exit": float(ep.loc[ep.exited, "dir_out"]
                                               .isin(["up_cont", "down_cont"]).mean()),
                    "post_ret_atr": S.boot_mean(ep.loc[ep.exited, "post_ret_atr"],
                                                ep.loc[ep.exited, "srv_date"], rng=rng),
                    "mfe_atr": float(ep["mfe_atr"].mean(skipna=True)),
                    "mae_atr": float(ep["mae_atr"].mean(skipna=True)),
                    "median_duration": float(ep["duration"].median()),
                    "t_peak_abn": float(ep["t_peak_abn"].mean()),
                },
                "taxonomy": ct.to_dict(),
                "by_depth": bucket_stats(ep, "depth_b", rng),
                "by_duration": bucket_stats(ep, "dur_b", rng),
                "by_session": bucket_stats(ep, "session", rng),
                "by_year": bucket_stats(ep, "year", rng),
            }
        report[sym] = res

        r = res["adj"]
        o = r["overall"]
        print(f"\n=== {sym} === (clock-standardised episodes, n={r['n_episodes']})")
        print(f"  forward 2h activity vs seasonal normal: {o['fwd_abn']['mean']:.3f} "
              f"CI[{o['fwd_abn']['ci_lo']:.3f},{o['fwd_abn']['ci_hi']:.3f}]  "
              f"| unconditional baseline {base['fwd_abn_uncond']:.3f}")
        m = r["spike_vs_matched"]
        if m:
            print(f"  P(clock-referenced spike): {m['observed']:.3f} vs "
                  f"tod x month matched {m['matched']:.3f}  lift={m['lift']:.3f} "
                  f"CI[{m['lift_ci_lo']:.3f},{m['lift_ci_hi']:.3f}] p={m['p_two_sided']:.3f}")
        print(f"  P(spike)={o['P_spike']:.3f}  P(gradual)={o['P_gradual']:.3f}  "
              f"P(stays low)={o['P_stays_low']:.3f}  P(exits box)={o['P_exit']:.3f}  "
              f"P(continue|exit)={o['P_cont_given_exit']:.3f}")
        pr = o["post_ret_atr"]
        print(f"  post-exit directional return: {pr['mean']:+.4f} ATR "
              f"CI[{pr['ci_lo']:+.4f},{pr['ci_hi']:+.4f}]   "
              f"MFE={o['mfe_atr']:.2f} MAE={o['mae_atr']:.2f}")
        print("  by DEPTH (min percentile reached in the episode):")
        for k, v in sorted(r["by_depth"].items()):
            print(f"    {k:12s} n={v['n']:>5} fwd_abn={v['fwd_abn']['mean']:.3f} "
                  f"P(spike)={v['P_spike']:.3f} P(cont|exit)={v['P_cont_given_exit']:.3f} "
                  f"post_ret={v['post_ret_atr']['mean']:+.4f}")
        print("  by DURATION (bars already compressed):")
        for k in ["1-3", "4-6", "7-12", "13-24", "25+"]:
            v = r["by_duration"].get(k)
            if v:
                print(f"    {k:12s} n={v['n']:>5} fwd_abn={v['fwd_abn']['mean']:.3f} "
                      f"P(spike)={v['P_spike']:.3f} P(cont|exit)={v['P_cont_given_exit']:.3f} "
                      f"post_ret={v['post_ret_atr']['mean']:+.4f} "
                      f"CI[{v['post_ret_atr']['ci_lo']:+.4f},{v['post_ret_atr']['ci_hi']:+.4f}]")

        RL.register(
            test_id=f"P2.postexit.{sym}.adj", part="2/7",
            hypothesis="Price continues in the direction of the break out of a compression episode",
            definition=f"mean signed return from first box exit to end of a {W}-bar window, in ATR units",
            instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
            n=o["post_ret_atr"]["n"],
            ci=(o["post_ret_atr"]["ci_lo"], o["post_ret_atr"]["ci_hi"]),
            control="day-block bootstrap; sign convention fixed by the break direction",
            result={"post_ret_atr": o["post_ret_atr"]["mean"],
                    "P_cont_given_exit": o["P_cont_given_exit"]})
        RL.register(
            test_id=f"P2.fwdabn.{sym}.adj", part="2",
            hypothesis="A compression episode ending is followed by above-normal activity",
            definition="sum(range) / sum(seasonal range) over the 24 bars after episode end",
            instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
            n=o["fwd_abn"]["n"], ci=(o["fwd_abn"]["ci_lo"], o["fwd_abn"]["ci_hi"]),
            control="seasonal expectation for the exact clock slots spanned",
            result={"fwd_abn": o["fwd_abn"]["mean"]})

    with (OUT / "r2_transitions.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r2_transitions.json")


if __name__ == "__main__":
    main()
