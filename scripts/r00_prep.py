"""Build a clean M5 panel from the FTMO export for the volatility-compression study.

Three traps this handles explicitly:

1. MT5 hands back *hourly* bars in the "M5" series for the deep-history region of
   some index symbols (US100/US500 before ~Oct-2021, GER40 for most of 2021).
   The bar spacing histogram proves it. Everything before 2022-01-01 is dropped
   so all symbols share one clean, genuinely-5-minute window.

2. Raw MT5 timestamps are the broker's *wall clock* (FTMO = EET/EEST), not UTC.
   Subtracting a fixed offset is right for ~48 weeks a year and one hour wrong
   for the rest, which smears every session bucket. We localize the wall clock
   through the tz database instead, then derive true UTC and New York time.

3. data/raw/specs.csv is column-shifted (the symbol field is missing on every
   row but GBPJPY), so point size is NOT read from it. Point size is asserted
   against the observed decimal precision of the price series instead.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RAW = Path(_DATA)
OUT = Path(_os.path.join(_REPO, "results"))
OUT.mkdir(parents=True, exist_ok=True)

# FTMO-Demo runs EET/EEST (UTC+2 winter, UTC+3 summer, EU DST rules).
SERVER_TZ = ZoneInfo("Europe/Bucharest")
NY_TZ = ZoneInfo("America/New_York")

START = "2022-01-01"

# asset_class is used for grouping in the report, not in any calculation.
SYMBOLS = {
    "US100.cash": dict(point=0.01, asset="index", apex="NQ"),
    "US500.cash": dict(point=0.01, asset="index", apex="ES"),
    "US30.cash":  dict(point=0.01, asset="index", apex="YM"),
    "GER40.cash": dict(point=0.01, asset="index", apex="FDAX"),
    "XAUUSD":     dict(point=0.01, asset="metal", apex="GC"),
    "EURUSD":     dict(point=0.00001, asset="fx",  apex="6E"),
}


def load_symbol(sym: str, tf: str = "M5") -> pd.DataFrame:
    src = RAW / f"bars_{sym}_{tf}.csv"
    df = pd.read_csv(src)

    # --- server wall clock -> true UTC -----------------------------------
    naive = pd.to_datetime(df["time"], unit="s")          # server wall clock
    srv = naive.dt.tz_localize(SERVER_TZ, ambiguous="NaT", nonexistent="NaT")
    bad = srv.isna().sum()
    df = df.loc[srv.notna()].copy()
    srv = srv.dropna()

    df["utc"] = srv.dt.tz_convert("UTC")
    df["srv"] = srv
    df = df.set_index("utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = df.loc[pd.Timestamp(START, tz="UTC"):]

    # --- integrity -------------------------------------------------------
    step = df.index.to_series().diff().dt.total_seconds()
    off_grid = int(((step % 300) != 0).sum())
    non_m5 = int(((step != 300) & (step.notna())).sum())
    assert (df["high"] >= df["low"]).all(), f"{sym}: high < low"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), f"{sym}: high < oc"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), f"{sym}: low > oc"

    # --- point size sanity check ----------------------------------------
    pt = SYMBOLS[sym]["point"]
    resid = np.abs(np.round(df["close"].to_numpy() / pt) - df["close"].to_numpy() / pt)
    assert resid.max() < 1e-3, f"{sym}: close not a multiple of point {pt}"

    ny = df.index.tz_convert(NY_TZ)
    df["ny_hour"] = ny.hour
    df["ny_minute"] = ny.minute
    df["ny_tod"] = ny.hour * 60 + ny.minute       # minutes since NY midnight
    df["ny_date"] = ny.normalize().tz_localize(None)
    df["dow"] = ny.dayofweek

    print(f"{sym:12s} bars={len(df):>7,}  {df.index[0]} -> {df.index[-1]}  "
          f"dropped_dst_nat={bad}  gaps(non-300s)={non_m5}  off_grid={off_grid}  "
          f"spread_med={df['spread'].median():.0f}pt")
    return df


def verify_clock(frames: dict[str, pd.DataFrame]) -> None:
    """The NY cash open is the loudest event in the tape. If our clock is right,
    tick volume for the index CFDs must jump exactly at NY 09:30 in *both*
    halves of the year -- that is the check a fixed GMT offset fails."""
    print("\n--- clock verification: mean tick_volume by NY time-of-day ---")
    for sym in ("US100.cash", "US500.cash"):
        df = frames[sym]
        for label, mask in (("Jan (EST)", df.index.month == 1),
                            ("Jul (EDT)", df.index.month == 7)):
            sub = df.loc[mask]
            prof = sub.groupby("ny_tod")["tick_volume"].mean()
            peak = prof.idxmax()
            v0925 = prof.get(9 * 60 + 25, np.nan)
            v0930 = prof.get(9 * 60 + 30, np.nan)
            print(f"  {sym:11s} {label}: peak at NY "
                  f"{peak // 60:02d}:{peak % 60:02d}   "
                  f"09:25={v0925:8.0f} -> 09:30={v0930:8.0f} "
                  f"(x{v0930 / v0925:.1f})")


def main() -> None:
    frames = {s: load_symbol(s) for s in SYMBOLS}
    verify_clock(frames)
    store = OUT / "panel_m5.pkl"
    pd.to_pickle({s: d for s, d in frames.items()}, store)
    print(f"\nwrote {store}")


if __name__ == "__main__":
    sys.exit(main())
