"""R0 -- causal feature panel.

Everything here is computed so that a bar at time t uses only information
available at the close of t. Two constructions deserve comment.

CLOCK-STANDARDISED VOLATILITY (Part 6). The first study showed that a raw ATR
percentile on M5 index CFDs is, to first order, a detector for "it is currently
the Asian session" -- 93-96% of compression bars on NQ/ES/YM fell in one session.
That makes "low ATR" and "this is normally a quiet time of day" the same
variable, and any hypothesis built on it inherits the clock.

So we estimate the seasonal volatility profile explicitly:

    S_tod(t) = median ATR14 observed in the SAME five-minute slot over the
               previous 60 sessions, shifted one session so today never
               contributes to its own expectation

    abn_t    = ATR14_t / S_tod(t)          "how quiet is this, for this clock time"

`abn` is the object the hypothesis should really be built on. abn = 0.5 at 03:00
means genuinely abnormal stillness; ATR = 0.5 x daily-median at 03:00 means
nothing at all, because 03:00 is always quiet.

The seasonal profile is keyed on NEW YORK time of day, not broker time. The
volatility U-shape is driven by information arrival -- the cash open, the 08:30
releases, the European open -- which is anchored to those exchanges' clocks, not
to the broker's bookkeeping boundary. The four weeks a year when US and EU
daylight saving disagree therefore add a little noise to the seasonal estimate,
which is the correct trade: the alternative smears the actual events.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

from pathlib import Path

import numpy as np
import pandas as pd

import vol_core as V

SRC = Path(_os.path.join(_REPO, "results", "panel_m5.pkl"))
OUT = Path(_os.path.join(_REPO, "results"))
OUT.mkdir(parents=True, exist_ok=True)

SEASON_DAYS = 60          # sessions in the seasonal median
SEASON_MIN = 20           # minimum before the seasonal estimate is trusted
PCT_LOOKBACK = 100        # same as study 1, so only the INPUT changes

POINT = {"US100.cash": 0.01, "US500.cash": 0.01, "US30.cash": 0.01,
         "GER40.cash": 0.01, "XAUUSD": 0.01, "EURUSD": 0.00001}


# --------------------------------------------------------------------------
def wilder_arr(x: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    v = x[:p]
    if np.isnan(v).any():
        v = np.nan_to_num(v)
    s = v.mean()
    out[p - 1] = s
    a, b = (p - 1) / p, 1 / p
    for i in range(p, len(x)):
        xi = x[i] if not np.isnan(x[i]) else 0.0
        s = a * s + b * xi
        out[i] = s
    return out


def adx_rsi(high, low, close, p=14):
    n = len(close)
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = wilder_arr(tr, p)
    pdi = 100 * wilder_arr(pdm, p) / np.where(atr == 0, np.nan, atr)
    ndi = 100 * wilder_arr(ndm, p) / np.where(atr == 0, np.nan, atr)
    dx = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) == 0, np.nan, pdi + ndi)
    adx = wilder_arr(np.nan_to_num(dx), p)

    diff = np.diff(close, prepend=close[0])
    gain = wilder_arr(np.where(diff > 0, diff, 0.0), p)
    loss = wilder_arr(np.where(diff < 0, -diff, 0.0), p)
    rs = gain / np.where(loss == 0, np.nan, loss)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.where(loss == 0, 100.0, rsi)
    return adx, rsi


def episode_id(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label maximal runs of True. Returns (episode id or -1, age within run)."""
    n = len(state)
    eid = np.full(n, -1, dtype=np.int64)
    age = np.zeros(n, dtype=np.int64)
    cur = -1
    for i in range(n):
        if state[i]:
            if i == 0 or not state[i - 1]:
                cur += 1
                age[i] = 1
            else:
                age[i] = age[i - 1] + 1
            eid[i] = cur
    return eid, age


# --------------------------------------------------------------------------
def build(sym: str, df: pd.DataFrame) -> pd.DataFrame:
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    opn = df["open"].to_numpy(np.float64)
    vol = df["tick_volume"].to_numpy(np.float64)
    epoch = (df.index.tz_convert("UTC").to_numpy()
             .astype("datetime64[s]").astype("int64"))
    tod = df["ny_tod"].to_numpy()
    gap = V.gap_flags(epoch)

    f = pd.DataFrame(index=df.index)
    f["sym"] = sym
    f["open"], f["high"], f["low"], f["close"] = opn, high, low, close
    f["spread_px"] = df["spread"].to_numpy(np.float64) * POINT[sym]
    f["tod"] = tod
    f["ny_date"] = df["ny_date"].to_numpy()
    f["dow"] = df["dow"].to_numpy()
    f["srv_date"] = df["srv"].dt.tz_localize(None).dt.normalize().to_numpy()
    f["month"] = pd.PeriodIndex(pd.DatetimeIndex(f["srv_date"]), freq="M").astype(str)
    f["year"] = pd.DatetimeIndex(f["srv_date"]).year
    f["gap"] = gap

    # --- volatility -----------------------------------------------------
    r = high - low
    f["R"] = r
    f["atr"] = V.wilder(r)
    f["pct_raw"] = V.pct_rank(f["atr"].to_numpy(), PCT_LOOKBACK)

    # seasonal profile: trailing median at the same 5-minute slot, shifted so
    # today's own value never enters its own expectation
    f["atr_seasonal"] = (f.groupby("tod", observed=True)["atr"]
                          .transform(lambda s: s.shift(1)
                                     .rolling(SEASON_DAYS, min_periods=SEASON_MIN)
                                     .median()))
    f["abn"] = f["atr"] / f["atr_seasonal"]
    f["pct_adj"] = V.pct_rank(f["abn"].to_numpy(), PCT_LOOKBACK)

    # --- trend / momentum / oscillators ---------------------------------
    adx, rsi = adx_rsi(high, low, close)
    f["adx"] = adx
    f["rsi"] = rsi
    lc = np.log(close)
    for k in (12, 48, 288):
        f[f"ret_{k}"] = np.concatenate([np.full(k, np.nan), lc[k:] - lc[:-k]])
    f["atr_slope"] = f["atr"] / f["atr"].shift(12) - 1.0
    ema = pd.Series(close, index=df.index).ewm(span=288, adjust=False).mean()
    f["htf_dist"] = (close - ema.to_numpy()) / f["atr"]
    f["htf_slope"] = (ema - ema.shift(288)) / f["atr"]

    # --- session anchors -------------------------------------------------
    g = f.groupby("srv_date", observed=True)
    f["sess_open"] = g["open"].transform("first")
    f["dist_sessopen"] = (f["close"] - f["sess_open"]) / f["atr"]
    pv = (f["close"] * vol).groupby(f["srv_date"]).cumsum()
    vv = pd.Series(vol, index=f.index).groupby(f["srv_date"]).cumsum()
    f["vwap"] = pv / vv.replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr"]

    day_hi = g["high"].max()
    day_lo = g["low"].min()
    f["pdh"] = f["srv_date"].map(day_hi.shift(1))
    f["pdl"] = f["srv_date"].map(day_lo.shift(1))
    f["dist_pdh"] = (f["close"] - f["pdh"]) / f["atr"]
    f["dist_pdl"] = (f["close"] - f["pdl"]) / f["atr"]

    box_hi = f["high"].rolling(12).max().shift(1)
    box_lo = f["low"].rolling(12).min().shift(1)
    f["box_hi"], f["box_lo"] = box_hi, box_lo
    f["pos_in_box"] = (f["close"] - box_lo) / (box_hi - box_lo).replace(0, np.nan)

    # --- sessions (New York clock) ---------------------------------------
    t = f["tod"].to_numpy()
    sess = np.full(len(f), "asia", dtype=object)
    sess[(t >= 3 * 60) & (t < 8 * 60)] = "london"
    sess[(t >= 8 * 60) & (t < 12 * 60)] = "ny_am"
    sess[(t >= 12 * 60) & (t < 17 * 60)] = "ny_pm"
    f["session"] = sess

    # --- volatility states, on BOTH definitions ---------------------------
    for tag, col in (("raw", "pct_raw"), ("adj", "pct_adj")):
        p = f[col].to_numpy()
        st = np.full(len(f), "na", dtype=object)
        with np.errstate(invalid="ignore"):
            st[p < 20] = "compressed"
            st[(p >= 20) & (p <= 80)] = "normal"
            st[(p > 80) & (p <= 95)] = "high"
            st[p > 95] = "extreme"
        f[f"state_{tag}"] = st
        eid, age = episode_id(st == "compressed")
        f[f"epi_{tag}"] = eid
        f[f"epi_age_{tag}"] = age
    return f


def main():
    panel = pd.read_pickle(SRC)
    for sym, df in panel.items():
        f = build(sym, df)
        f.to_pickle(OUT / f"feat_{sym}.pkl")
        n = len(f)
        cov = f["abn"].notna().mean()
        print(f"{sym:12s} rows={n:>7,}  seasonal coverage={cov:.3f}  "
              f"compressed(raw)={(f['state_raw']=='compressed').mean():.3f}  "
              f"compressed(adj)={(f['state_adj']=='compressed').mean():.3f}  "
              f"episodes raw={f['epi_raw'].max()+1:>5}  adj={f['epi_adj'].max()+1:>5}")
        # how much does the clock adjustment change WHICH bars are selected?
        a = (f["state_raw"] == "compressed").to_numpy()
        b = (f["state_adj"] == "compressed").to_numpy()
        ok = f["pct_raw"].notna().to_numpy() & f["pct_adj"].notna().to_numpy()
        print(f"{'':12s} overlap: {np.mean(a[ok] & b[ok]) / np.mean(a[ok] | b[ok]):.3f} Jaccard; "
              f"adj-compression in Asia={np.mean(f['session'][b & ok]=='asia'):.3f} "
              f"vs raw={np.mean(f['session'][a & ok]=='asia'):.3f}")


if __name__ == "__main__":
    main()
