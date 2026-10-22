"""R1 -- Part 1: are volatility states persistent regimes?

Reported for both state definitions (raw ATR percentile, and the
clock-standardised one), for all four states, at k = 1, 3, 6, 12, 24, 48.

Three comparisons, because a raw persistence percentage on its own says
nothing:

  * against the UNCONDITIONAL rate  -- is knowing the state worth anything?
  * against a FIRST-ORDER MARKOV chain fitted to the same data, evaluated as
    P^k -- if the k-step behaviour matches P^k the process is memoryless given
    the current state, and there is no extra information in how long you have
    been there. Deviation is what "regime" actually means.
  * against an IID surrogate (permuted range series) -- the floor.

Uncertainty is a day-block bootstrap throughout.
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
KS = [1, 3, 6, 12, 24, 48]
STATES = ["compressed", "normal", "high", "extreme"]
SYMS = ["XAUUSD", "EURUSD", "US100.cash", "US500.cash", "US30.cash", "GER40.cash"]
N_SURR = 100


def transition_matrix(st: np.ndarray, valid: np.ndarray, k: int) -> pd.DataFrame:
    n = len(st)
    a = st[:n - k]
    b = st[k:]
    m = valid[:n - k] & valid[k:]
    tab = pd.crosstab(pd.Series(a[m], name="from"), pd.Series(b[m], name="to"))
    tab = tab.reindex(index=STATES, columns=STATES, fill_value=0)
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)


def main():
    rng = np.random.default_rng(11)
    report = {}

    for sym in SYMS:
        f = pd.read_pickle(OUT / f"feat_{sym}.pkl")
        days = f["srv_date"].to_numpy()
        gap = f["gap"].to_numpy()
        res = {}

        for tag in ("raw", "adj"):
            st = f[f"state_{tag}"].to_numpy()
            base_ok = (st != "na")
            if tag == "adj":
                base_ok &= f["abn"].notna().to_numpy()

            uncond = {s: float(np.mean(st[base_ok] == s)) for s in STATES}

            # one-step matrix, then its k-th power = the memoryless prediction
            P1 = transition_matrix(st, base_ok, 1).fillna(0).to_numpy()

            block = {"unconditional": uncond, "P1": P1.tolist(), "k": {}}
            for k in KS:
                contig = V.contiguous_forward(gap, k)
                ok = base_ok & contig & np.roll(base_ok, -k)
                ok[-k:] = False
                Pk_obs = transition_matrix(st, base_ok & contig, k)
                Pk_markov = np.linalg.matrix_power(P1, k)

                ent = {}
                for i, s in enumerate(STATES):
                    cond = ok & (st == s)
                    if cond.sum() < 100:
                        continue
                    fut = np.roll(st, -k)
                    same = (fut == s).astype(float)
                    bs = S.boot_mean(same[cond], days[cond], rng=rng)
                    ent[s] = {
                        "P_same": bs["mean"], "ci": [bs["ci_lo"], bs["ci_hi"]],
                        "n": bs["n"], "n_days": bs["n_days"],
                        "unconditional": uncond[s],
                        "markov_k_step": float(Pk_markov[i, i]),
                        "lift_vs_unconditional": bs["mean"] / uncond[s] if uncond[s] else np.nan,
                        "excess_vs_markov": bs["mean"] - float(Pk_markov[i, i]),
                    }
                block["k"][k] = {"per_state": ent,
                                 "observed_matrix": Pk_obs.fillna(0).to_numpy().tolist(),
                                 "markov_matrix": Pk_markov.tolist()}
            res[tag] = block

        # --- iid surrogate floor, raw definition only --------------------
        r = f["R"].to_numpy(np.float64)
        surr = {k: [] for k in KS}
        for _ in range(N_SURR):
            rs = rng.permutation(r)
            atr = V.wilder(rs)
            p = V.pct_rank(atr, 100)
            s2 = np.where(np.isnan(p), "na",
                          np.where(p < 20, "compressed",
                                   np.where(p <= 80, "normal",
                                            np.where(p <= 95, "high", "extreme"))))
            ok0 = s2 != "na"
            for k in KS:
                ok = ok0 & V.contiguous_forward(gap, k)
                ok[-k:] = False
                c = ok & (s2 == "compressed")
                surr[k].append(float(np.mean(np.roll(s2, -k)[c] == "compressed")))
        res["iid_surrogate_compressed"] = {
            str(k): {"mean": float(np.mean(v)),
                     "p2.5": float(np.percentile(v, 2.5)),
                     "p97.5": float(np.percentile(v, 97.5))}
            for k, v in surr.items()}

        report[sym] = res

        print(f"\n=== {sym} ===")
        for tag in ("raw", "adj"):
            print(f"  --- {tag} state definition ---")
            print(f"    {'state':11s} {'uncond':>7}" +
                  "".join(f"{'k='+str(k):>19}" for k in KS))
            for s in STATES:
                row = f"    {s:11s} {res[tag]['unconditional'][s]:>7.3f}"
                for k in KS:
                    e = res[tag]["k"][k]["per_state"].get(s)
                    row += (f"  {e['P_same']:.3f}[{e['ci'][0]:.3f},{e['ci'][1]:.3f}]"
                            if e else f"{'--':>19}")
                print(row)
            print(f"    excess over first-order Markov (compressed): " +
                  "  ".join(f"k={k}:{res[tag]['k'][k]['per_state']['compressed']['excess_vs_markov']:+.3f}"
                            for k in KS if "compressed" in res[tag]["k"][k]["per_state"]))
        sr = res["iid_surrogate_compressed"]
        print("    iid surrogate P(still compressed): " +
              "  ".join(f"k={k}:{sr[str(k)]['mean']:.3f}" for k in KS))

        for tag in ("raw", "adj"):
            for k in (12, 48):
                e = res[tag]["k"][k]["per_state"].get("compressed")
                if e:
                    RL.register(
                        test_id=f"P1.persist.{sym}.{tag}.k{k}", part="1",
                        hypothesis="Volatility states are persistent regimes, not memoryless draws",
                        definition=f"P(compressed at t+{k} | compressed at t), {tag} state definition",
                        instrument=sym, sample="M5 2022-01..2026-07",
                        prespecified=True, n=e["n"],
                        ci=(e["ci"][0], e["ci"][1]),
                        control="unconditional rate + first-order Markov P^k + iid surrogate",
                        result={"P_same": e["P_same"], "unconditional": e["unconditional"],
                                "markov": e["markov_k_step"],
                                "excess_vs_markov": e["excess_vs_markov"]},
                        notes="descriptive estimate, not a directional edge claim")

    with (OUT / "r1_persistence.json").open("w") as fh:
        json.dump(report, fh, indent=1, default=float)
    RL.flush()
    print("\nwrote", OUT / "r1_persistence.json")


if __name__ == "__main__":
    main()
