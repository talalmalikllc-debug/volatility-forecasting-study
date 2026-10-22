"""Shared primitives for the volatility-compression study.

Everything downstream is a deterministic function of ONE series: the per-bar
range R_t = High_t - Low_t. That is a deliberate design choice, not laziness:

  * True Range would fold the overnight/maintenance gap into the bar that
    follows it, which for index CFDs injects a fake volatility spike at the same
    clock time every single day.
  * More importantly, because the whole pipeline (ATR -> rolling percentile ->
    state -> expansion test) is a pure function of {R_t}, a surrogate series
    built by *permuting* {R_t} passes through byte-identical machinery. Any
    difference between the real result and the surrogate result is therefore
    attributable to temporal structure in volatility and to nothing else --
    not to the smoothing, not to the bounded percentile window, not to the
    marginal distribution of ranges.

A True-Range variant is available via `ranges(..., use_tr=True)` and is run as a
robustness check.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ATR_P = 14
LOOKBACK = 100
BAR_SEC = 300


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def ranges(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           use_tr: bool = False) -> np.ndarray:
    if not use_tr:
        return high - low
    pc = np.roll(close, 1)
    pc[0] = close[0]
    return np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))


def wilder(r: np.ndarray, p: int = ATR_P) -> np.ndarray:
    """Wilder's smoothed average. NaN until the seed is complete."""
    out = np.full(r.shape, np.nan, dtype=np.float64)
    a, b = (p - 1) / p, 1 / p
    s = r[:p].mean()
    out[p - 1] = s
    for i in range(p, len(r)):
        s = a * s + b * r[i]
        out[i] = s
    return out


def pct_rank(x: np.ndarray, lookback: int = LOOKBACK,
             warmup: int = ATR_P - 1) -> np.ndarray:
    """Rolling percentile rank of x_t inside {x_{t-lookback+1} .. x_t}, in %.

    Causal by construction: the window ends at t, so no future bar is consulted.
    """
    n = len(x)
    out = np.full(n, np.nan)
    first = warmup + lookback - 1
    if n <= first:
        return out
    w = sliding_window_view(x, lookback)          # w[i] = x[i : i+lookback]
    w = w[warmup:]                                # drop windows touching NaNs
    cur = x[first:]
    out[first:] = (w <= cur[:, None]).sum(axis=1) / lookback * 100.0
    return out


# --------------------------------------------------------------------------
# forward-window helpers  (out[t] refers strictly to bars t+1 .. t+k)
# --------------------------------------------------------------------------
def fwd_max(a: np.ndarray, k: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan)
    if n <= k:
        return out
    m = sliding_window_view(a, k).max(axis=1)     # m[i] = max(a[i : i+k])
    out[:n - k] = m[1:]
    return out


def fwd_sum(a: np.ndarray, k: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan)
    cs = np.concatenate([[0.0], np.nancumsum(a)])
    out[:n - k] = cs[k + 1:] - cs[1:n - k + 1]
    return out


def gap_flags(epoch: np.ndarray) -> np.ndarray:
    """gap[i] is True when bar i does not sit exactly one bar after bar i-1."""
    step = np.diff(epoch, prepend=epoch[0] - BAR_SEC)
    return step != BAR_SEC


def contiguous_forward(gap: np.ndarray, k: int) -> np.ndarray:
    """True at t when bars t+1 .. t+k are an unbroken run of 5-minute bars."""
    n = len(gap)
    cg = np.concatenate([[0], np.cumsum(gap.astype(np.int64))])
    out = np.zeros(n, dtype=bool)
    idx = np.arange(n - k)
    out[:n - k] = (cg[idx + k + 1] - cg[idx + 1]) == 0
    return out


def contiguous_back(gap: np.ndarray, k: int) -> np.ndarray:
    """True at t when bars t-k .. t are an unbroken run."""
    n = len(gap)
    cg = np.concatenate([[0], np.cumsum(gap.astype(np.int64))])
    out = np.zeros(n, dtype=bool)
    idx = np.arange(k, n)
    out[k:] = (cg[idx + 1] - cg[idx - k + 1]) == 0
    return out


# --------------------------------------------------------------------------
# surrogates
# --------------------------------------------------------------------------
def surrogate_iid(r: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Null 1 -- destroys ALL temporal structure, keeps the exact marginal."""
    return rng.permutation(r)


def surrogate_tod(r: np.ndarray, tod: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Null 2 -- permutes ranges only *within* each time-of-day bucket.

    Keeps the intraday volatility seasonality (the U-shape, the 09:30 explosion,
    the dead 03:00 tape) perfectly intact while destroying every bar-to-bar and
    day-to-day dependency. This is the null that matters: beating it means
    compression carries information the wall clock does not already give you.
    """
    out = np.empty_like(r)
    order = np.argsort(tod, kind="stable")
    grouped = tod[order]
    bounds = np.flatnonzero(np.diff(grouped)) + 1
    for sl in np.split(order, bounds):
        out[sl] = rng.permutation(r[sl])
    return out
