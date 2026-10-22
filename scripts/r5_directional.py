"""R5 -- Parts 7, 8, 9, 10: is there ANY directional information?

Everything found so far is a volatility statement. A CFD account cannot trade
volatility directly, so this is where a tradable phenomenon would have to live.

EVENT (strictly causal, the fill is on the next bar's open):

    bar j is an expansion bar emerging from compression when
        pct_adj[j-1] < 20                      (abnormally quiet for the clock)
        range_j / seasonal_range_j >= 2.0      (a clock-referenced impulse)
    direction d = sign(close_j - open_j)
    entry = open_{j+1}

Returns are expressed in units of the IMPULSE BAR'S OWN RANGE, R_j. That makes
them comparable across instruments and regimes, and it puts the transaction cost
on the same axis: a spread of 0.15 R means the move has to travel 15% of the
impulse bar just to break even.

PART 7 asks the sign question directly: continuation (>0), reversal (<0), or
noise. No matching is needed for the sign, because d flips the convention
symmetrically -- but the unsigned drift is reported alongside so that a
persistent long bias in the instrument cannot masquerade as an edge.

PARTS 8-10 screen each state variable INDEPENDENTLY, as instructed. For each,
events are split into quintiles and the q5-q1 difference in the mean signed
return is bootstrapped over days. One test per (variable, instrument) at the
primary horizon, so the multiple-testing accounting stays tractable; survivors
are then re-examined across horizons, years and instruments.
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
import vol_core as V

OUT = Path(_os.path.join(_REPO, "results"))
SYMS = ["XAUUSD", "EURUSD", "US100.cash", "US500.cash", "US30.cash", "GER40.cash"]
HORIZONS = [1, 3, 6, 12, 24]
K_PRIMARY = 12
IMPULSE = 2.0


def build_events(f: pd.DataFrame):
    n = len(f)
    R = f["R"].to_numpy(); seas = f["atr_seasonal"].to_numpy()
    close = f["close"].to_numpy(); opn = f["open"].to_numpy()
    high = f["high"].to_numpy(); low = f["low"].to_numpy()
    gap = f["gap"].to_numpy()
    pct_adj = f["pct_adj"].to_numpy()

    impulse = R / seas
    prev_comp = np.roll(pct_adj, 1) < 20
    body = close - opn
    is_ev = prev_comp & (impulse >= IMPULSE) & (body != 0) & ~np.isnan(impulse)
    is_ev &= ~np.roll(gap, -1)                 # bar j+1 must follow j
    is_ev[:400] = False
    is_ev[-(max(HORIZONS) + 3):] = False
    idx = np.flatnonzero(is_ev)
    idx = idx[V.contiguous_forward(gap, max(HORIZONS) + 1)[idx]]
    if len(idx) < 200:
        return None

    d = np.sign(body[idx])
    entry = opn[idx + 1]
    Rj = R[idx]

    ev = {"idx": idx, "d": d, "Rj": Rj,
          "day": f["srv_date"].to_numpy()[idx],
          "year": f["year"].to_numpy()[idx],
          "month": f["month"].to_numpy()[idx],
          "tod": f["tod"].to_numpy()[idx],
          "session": f["session"].to_numpy()[idx],
          "spread_R": f["spread_px"].to_numpy()[idx + 1] / Rj}
    for k in HORIZONS:
        ev[f"sr{k}"] = d * (close[idx + k] - entry) / Rj      # signed by the impulse
        ev[f"ur{k}"] = (close[idx + k] - entry) / Rj          # unsigned, for drift
        hs = np.array([high[i + 1:i + 1 + k].max() for i in idx])
        ls = np.array([low[i + 1:i + 1 + k].min() for i in idx])
        fav = np.where(d > 0, hs, -ls)
        adv = np.where(d > 0, ls, -hs)
        ev[f"mfe{k}"] = (fav - d * entry) / Rj
        ev[f"mae{k}"] = (d * entry - adv) / Rj

    prev = idx - 1
    def sgn(col):
        return d * f[col].to_numpy()[idx]
    pos = f["pos_in_box"].to_numpy()[prev]
    ev["v_mom12"] = sgn("ret_12")
    ev["v_mom48"] = sgn("ret_48")
    ev["v_mom288"] = sgn("ret_288")
    ev["v_htf_dist"] = sgn("htf_dist")
    ev["v_htf_slope"] = sgn("htf_slope")
    ev["v_vwap_dist"] = sgn("vwap_dist")
    ev["v_sessopen"] = sgn("dist_sessopen")
    ev["v_pdh"] = sgn("dist_pdh")
    ev["v_pdl"] = sgn("dist_pdl")
    ev["v_pos_in_box"] = np.where(d > 0, pos, 1 - pos)
    ev["v_adx"] = f["adx"].to_numpy()[prev]
    ev["v_rsi_aligned"] = np.where(d > 0, f["rsi"].to_numpy()[prev],
                                   100 - f["rsi"].to_numpy()[prev])
    ev["v_atr_slope"] = f["atr_slope"].to_numpy()[prev]
    ev["v_abn_prev"] = f["abn"].to_numpy()[prev]
    ev["v_impulse"] = impulse[idx]
    ev["v_body_frac"] = np.abs(body[idx]) / Rj
    ev["v_close_loc"] = np.where(d > 0, (close[idx] - low[idx]) / Rj,
                                 (high[idx] - close[idx]) / Rj)
    return pd.DataFrame(ev)


VARS = [c for c in ("v_mom12 v_mom48 v_mom288 v_htf_dist v_htf_slope v_vwap_dist "
                    "v_sessopen v_pdh v_pdl v_pos_in_box v_adx v_rsi_aligned "
                    "v_atr_slope v_abn_prev v_impulse v_body_frac v_close_loc").split()]


def main():
    rng = np.random.default_rng(59)
    report = {}

    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        e = build_events(f)
        if e is None:
            print(f"{sym}: too few events")
            continue
        e.to_pickle(OUT / f"events_{sym}.pkl")
        res = {"n_events": int(len(e)), "n_days": int(e["day"].nunique()),
               "mean_spread_R": float(e["spread_R"].mean())}

        # ---- PART 7: continuation or reversal? --------------------------
        p7 = {}
        for k in HORIZONS:
            b = S.boot_mean(e[f"sr{k}"], e["day"], rng=rng)
            drift = S.boot_mean(e[f"ur{k}"], e["day"], rng=rng)
            p7[k] = {"signed": b, "unsigned_drift": drift,
                     "mfe": float(e[f"mfe{k}"].mean()),
                     "mae": float(e[f"mae{k}"].mean())}
            RL.register(
                test_id=f"P7.signed.{sym}.k{k}", part="7",
                hypothesis="An expansion bar out of compression is followed by continuation",
                definition=f"mean signed return over {k} bars from next-bar open, in impulse-range units",
                instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                n=b["n"], ci=(b["ci_lo"], b["ci_hi"]),
                p_value=2 * min(b["p_gt0"], 1 - b["p_gt0"]),
                control="sign convention set by the impulse bar; unsigned drift reported alongside",
                result={"signed_mean_R": b["mean"], "drift": drift["mean"]})
        res["part7"] = p7

        # ---- PARTS 8-10: independent variable screen --------------------
        p8 = {}
        for v in VARS:
            x = e[v].to_numpy(dtype=float)
            ok = np.isfinite(x)
            if ok.sum() < 500:
                continue
            q = pd.qcut(pd.Series(x[ok]), 5, labels=False, duplicates="drop")
            sub = e.loc[ok].assign(_q=q.to_numpy())
            if sub["_q"].nunique() < 5:
                continue
            means = {}
            for qi, g in sub.groupby("_q", observed=True):
                bb = S.boot_mean(g[f"sr{K_PRIMARY}"], g["day"], rng=rng)
                means[int(qi)] = bb
            a = sub[sub["_q"] == 4]; b_ = sub[sub["_q"] == 0]
            dd = S.boot_diff(a[f"sr{K_PRIMARY}"], a["day"],
                             b_[f"sr{K_PRIMARY}"], b_["day"], rng=rng)
            p8[v] = {"quintile_means": means, "q5_minus_q1": dd}
            RL.register(
                test_id=f"P8.{sym}.{v}", part="8/9/10",
                hypothesis=f"{v} conditions the directional outcome of an expansion out of compression",
                definition=f"difference in mean signed {K_PRIMARY}-bar return between top and bottom quintile of {v}",
                instrument=sym, sample="M5 2022-01..2026-07", prespecified=True,
                n=dd["n_a"] + dd["n_b"], p_value=dd["p_two_sided"],
                ci=(dd["ci_lo"], dd["ci_hi"]),
                control="quintiles within the event set; day-block bootstrap of the difference",
                result={"q5_minus_q1_R": dd["diff"]})

        # ---- session conditioning (Part 9) ------------------------------
        sess = {}
        for s, g in e.groupby("session", observed=True):
            if len(g) < 200:
                continue
            sess[s] = S.boot_mean(g[f"sr{K_PRIMARY}"], g["day"], rng=rng)
        res["part8_variables"] = p8
        res["part9_session"] = sess
        report[sym] = res

        print(f"\n=== {sym} ===  n_events={len(e)} over {e['day'].nunique()} days, "
              f"mean spread = {e['spread_R'].mean():.3f} R")
        print("  PART 7 -- signed return after an expansion out of compression (units of impulse range)")
        for k in HORIZONS:
            b = p7[k]["signed"]; dr = p7[k]["unsigned_drift"]
            print(f"    k={k:>2}  {b['mean']:+.4f} R  CI[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]"
                  f"   drift={dr['mean']:+.4f}   MFE={p7[k]['mfe']:.3f} MAE={p7[k]['mae']:.3f}")
        print("  PARTS 8-10 -- q5 minus q1 of the signed 12-bar return, by variable")
        rows = sorted(p8.items(), key=lambda kv: kv[1]["q5_minus_q1"]["p_two_sided"])
        for v, d_ in rows:
            q = d_["q5_minus_q1"]
            star = "  <<<" if q["p_two_sided"] < 0.05 else ""
            print(f"    {v:16s} q5-q1={q['diff']:+.4f} R  "
                  f"CI[{q['ci_lo']:+.4f},{q['ci_hi']:+.4f}]  p={q['p_two_sided']:.3f}{star}")
        print("  session:", "  ".join(
            f"{s}={v['mean']:+.4f}[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]" for s, v in sess.items()))

    with (OUT / "r5_directional.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r5_directional.json")


if __name__ == "__main__":
    main()
