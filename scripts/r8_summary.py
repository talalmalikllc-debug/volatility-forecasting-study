"""R8 -- Part 13: multiple-testing accounting over the whole research tree."""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import research_log as RL

OUT = Path(_os.path.join(_REPO, "results"))


def main():
    log = json.load((OUT / "research_log.json").open())
    ents = log["entries"]
    pre = [e for e in ents if e["prespecified"] and e.get("p_value") is not None]
    exp = [e for e in ents if not e["prespecified"]]

    print(f"total registered tests : {len(ents)}")
    print(f"  pre-specified with p : {len(pre)}")
    print(f"  exploratory          : {len(exp)}")

    by_part = defaultdict(int)
    for e in ents:
        by_part[e["part"]] += 1
    print("  by part:", dict(sorted(by_part.items())))

    ps = [e["p_value"] for e in pre]
    passed = RL.benjamini_hochberg(ps, q=0.10)
    n_raw = sum(p < 0.05 for p in ps)
    print(f"\nraw p<0.05           : {n_raw} / {len(ps)}  "
          f"(chance expectation {0.05*len(ps):.1f})")
    print(f"survive BH FDR q=0.10: {sum(passed)} / {len(ps)}")

    surv = defaultdict(list)
    for e, ok in zip(pre, passed):
        if ok:
            surv[e["part"]].append(e)
    print("\nsurvivors by part:")
    for part, lst in sorted(surv.items()):
        print(f"  part {part:8s} {len(lst):>3} tests")

    # the directional family, isolated
    dirfam = [e for e in pre if e["part"] in ("7", "8/9/10")]
    dps = [e["p_value"] for e in dirfam]
    dpass = RL.benjamini_hochberg(dps, q=0.10) if dps else []
    print(f"\nDIRECTIONAL family (parts 7 + 8/9/10): {len(dps)} tests, "
          f"{sum(p < 0.05 for p in dps)} raw p<0.05 "
          f"(chance {0.05*len(dps):.1f}), {sum(dpass)} survive BH q=0.10")
    for e, ok in zip(dirfam, dpass):
        if ok:
            print(f"    SURVIVES: {e['test_id']}  p={e['p_value']:.4f}")

    # cross-instrument consistency for the directional screen
    byvar = defaultdict(list)
    for e in pre:
        if e["part"] == "8/9/10":
            v = e["test_id"].rsplit(".", 1)[-1]
            byvar[v].append((e["instrument"], e["result"]["q5_minus_q1_R"],
                             e["p_value"]))
    print("\ncross-instrument sign agreement in the directional screen "
          "(6 instruments each):")
    rows = []
    for v, lst in byvar.items():
        signs = [np.sign(x[1]) for x in lst]
        agree = max(signs.count(1), signs.count(-1))
        nsig = sum(p < 0.05 for _, _, p in lst)
        rows.append((agree, nsig, v, lst))
    for agree, nsig, v, lst in sorted(rows, reverse=True):
        print(f"    {v:16s} same-sign {agree}/{len(lst)}  raw p<0.05 on {nsig}")

    # volatility family
    volfam = [e for e in pre if e["part"] in ("5", "11/12", "4", "6", "2")]
    vps = [e["p_value"] for e in volfam]
    vpass = RL.benjamini_hochberg(vps, q=0.10) if vps else []
    print(f"\nVOLATILITY family: {len(vps)} tests, {sum(vpass)} survive BH q=0.10")

    json.dump({"n_tests": len(ents), "n_prespecified_with_p": len(pre),
               "n_exploratory": len(exp), "n_raw_sig": n_raw,
               "n_bh_survive": int(sum(passed)),
               "directional_family": {"n": len(dps),
                                      "n_raw_sig": int(sum(p < 0.05 for p in dps)),
                                      "n_bh_survive": int(sum(dpass))},
               "volatility_family": {"n": len(vps), "n_bh_survive": int(sum(vpass))}},
              (OUT / "r8_summary.json").open("w"), indent=1)
    print("\nwrote", OUT / "r8_summary.json")


if __name__ == "__main__":
    main()
