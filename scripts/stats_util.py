"""Shared estimation helpers.

Two rules are enforced everywhere in this study:

1. Standard errors come from a DAY-BLOCK bootstrap, never from a binomial
   formula. Forward windows overlap and volatility states run in long streaks,
   so bar-level independence overstates precision by an order of magnitude.

2. Any conditional probability that is going to be compared against a baseline
   is compared against a MATCHED baseline -- same five-minute slot, same
   calendar month -- because the first study showed that pooling across the day
   and across a drifting volatility regime manufactures effects out of nothing
   (a pooled lift of 1.58 sitting above every single per-year lift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 3000


def _day_index(days):
    u, inv = np.unique(days, return_inverse=True)
    return u, inv


def boot_mean(vals, days, n_boot=N_BOOT, rng=None):
    """Day-block bootstrap of a mean."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return dict(mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_gt0=np.nan, n=0, n_days=0)
    rng = rng or np.random.default_rng(0)
    u, inv = _day_index(days)
    s = np.bincount(inv, weights=vals, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), size=(n_boot, len(u)))
    m = s[pick].sum(1) / c[pick].sum(1)
    return dict(mean=float(vals.mean()), ci_lo=float(np.percentile(m, 2.5)),
                ci_hi=float(np.percentile(m, 97.5)),
                p_gt0=float((m > 0).mean()), n=int(len(vals)), n_days=int(len(u)))


def boot_diff(a_vals, a_days, b_vals, b_days, n_boot=N_BOOT, rng=None):
    """Day-block bootstrap of (mean A - mean B), resampling each group's days."""
    rng = rng or np.random.default_rng(1)
    out = []
    for v, d in ((a_vals, a_days), (b_vals, b_days)):
        v = np.asarray(v, dtype=float)
        u, inv = _day_index(d)
        s = np.bincount(inv, weights=v, minlength=len(u))
        c = np.bincount(inv, minlength=len(u)).astype(float)
        pick = rng.integers(0, len(u), size=(n_boot, len(u)))
        out.append(s[pick].sum(1) / c[pick].sum(1))
    d_ = out[0] - out[1]
    obs = float(np.mean(a_vals) - np.mean(b_vals))
    p_two = 2 * min(float((d_ > 0).mean()), float((d_ < 0).mean()))
    return dict(diff=obs, ci_lo=float(np.percentile(d_, 2.5)),
                ci_hi=float(np.percentile(d_, 97.5)),
                p_two_sided=min(1.0, p_two),
                n_a=int(len(a_vals)), n_b=int(len(b_vals)))


def matched_baseline(event, valid, cond, tod, month, days, min_cell=20,
                     n_boot=N_BOOT, rng=None):
    """Direct standardisation on (5-min slot x calendar month).

    Each conditioned observation is paired with the rate ordinary bars achieved
    in the same slot of the same month; the lift is then bootstrapped over days
    with the pair kept together, so numerator and denominator move as one.
    """
    rng = rng or np.random.default_rng(2)
    d = pd.DataFrame({"tod": tod, "month": month, "ev": np.asarray(event, float),
                      "valid": valid, "cond": cond, "day": days})
    ok = d[d.valid]
    cell = ok.groupby(["tod", "month"], observed=True)["ev"].agg(["mean", "size"])
    cell = cell[cell["size"] >= min_cell]
    ev = d[d.cond & d.valid].merge(cell, left_on=["tod", "month"],
                                   right_index=True, how="inner")
    if len(ev) < 50:
        return None
    u, inv = _day_index(ev["day"].to_numpy())
    so = np.bincount(inv, weights=ev["ev"].to_numpy(), minlength=len(u))
    se = np.bincount(inv, weights=ev["mean"].to_numpy(), minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), size=(n_boot, len(u)))
    obs_b = so[pick].sum(1) / c[pick].sum(1)
    exp_b = se[pick].sum(1) / c[pick].sum(1)
    lift = obs_b / exp_b
    diff = obs_b - exp_b
    obs = float(ev["ev"].mean())
    exp = float(ev["mean"].mean())
    p_two = 2 * min(float((diff > 0).mean()), float((diff < 0).mean()))
    return dict(n=int(len(ev)), n_days=int(len(u)), observed=obs,
                matched=exp, lift=obs / exp if exp else np.nan,
                diff=obs - exp,
                lift_ci_lo=float(np.percentile(lift, 2.5)),
                lift_ci_hi=float(np.percentile(lift, 97.5)),
                diff_ci_lo=float(np.percentile(diff, 2.5)),
                diff_ci_hi=float(np.percentile(diff, 97.5)),
                p_two_sided=min(1.0, p_two))
