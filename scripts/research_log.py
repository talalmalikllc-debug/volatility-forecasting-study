"""Research log -- Part 13.

Every hypothesis test in this study registers itself here BEFORE its result is
looked at, with a flag saying whether it was pre-specified in the research plan
or discovered by looking at the data. At the end the whole tree is written out,
including everything that failed, and Benjamini-Hochberg is applied across the
pre-specified family so the surviving p-values mean something.

The point is not bookkeeping for its own sake. It is that "we ran 400 tests and
this one had p=0.01" and "we ran one test and it had p=0.01" are completely
different claims, and only a log written as you go can tell them apart.
"""
from __future__ import annotations
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("BROKER_DATA_DIR", _os.path.join(_REPO, "data", "raw"))

import json
from pathlib import Path

OUT = Path(_os.path.join(_REPO, "results"))
LOG_PATH = OUT / "research_log.json"

_entries: list[dict] = []


def register(*, test_id: str, part: str, hypothesis: str, definition: str,
             instrument: str, sample: str, prespecified: bool,
             result: dict, p_value: float | None = None,
             ci: tuple | None = None, n: int | None = None,
             control: str = "", notes: str = "") -> dict:
    e = {
        "test_id": test_id, "part": part, "hypothesis": hypothesis,
        "definition": definition, "instrument": instrument, "sample": sample,
        "prespecified": prespecified, "control": control,
        "n": n, "p_value": p_value, "ci": list(ci) if ci else None,
        "result": result, "notes": notes,
    }
    _entries.append(e)
    return e


def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Return, for each p-value, whether it is rejected at FDR level q."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    passed = [False] * m
    kmax = -1
    for rank, i in enumerate(idx, start=1):
        if pvals[i] <= q * rank / m:
            kmax = rank
    for rank, i in enumerate(idx, start=1):
        if rank <= kmax:
            passed[i] = True
    return passed


def flush(extra: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"n_tests": len(_entries), "entries": _entries}
    if extra:
        payload.update(extra)
    existing = []
    if LOG_PATH.exists():
        with LOG_PATH.open() as fh:
            existing = json.load(fh).get("entries", [])
    seen = {e["test_id"] for e in _entries}
    merged = [e for e in existing if e["test_id"] not in seen] + _entries
    payload["entries"] = merged
    payload["n_tests"] = len(merged)
    with LOG_PATH.open("w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(f"[research log] {len(_entries)} new, {len(merged)} total -> {LOG_PATH}")
