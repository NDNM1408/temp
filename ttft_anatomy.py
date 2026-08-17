#!/usr/bin/env python3
"""Where do the TTFT points actually go, per request?

The aggregate score says half the TTFT budget is missing, but not why, and the
two candidate causes want opposite fixes. A request can be slow because it had
to prefill tokens the cache did not hold -- that is a capacity problem, fixed by
holding more -- or because it waited behind somebody else's prefill, which is a
scheduling problem and gets worse, not better, when you make the cache bigger.

Both show up as one number in the export, so this splits them. Every record
carries the tokens the server read from cache, so `isl - cached` is the work
that request actually had to do. Dividing by the best prefill rate anyone
achieved in the run gives a floor on how long it could possibly have taken; what
is left over is time the request spent not being served.

Points are attributed the same way the scorer computes them, so the per-bucket
"lost" column sums to the gap between the measured score and a perfect 50.

Usage:
    python3 ttft_anatomy.py /lab/bench/base_v3
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

F_TTFT, C_TTFT, GAMMA, W = 200.0, 6000.0, 2.0, 0.5
BUCKETS = [(0, 1_000), (1_000, 4_000), (4_000, 16_000),
           (16_000, 64_000), (64_000, 10**9)]


def s_ttft(x: float) -> float:
    v = (C_TTFT - x) / (C_TTFT - F_TTFT)
    return (0.0 if v < 0 else min(v, 1.0)) ** GAMMA


def mv(r: dict, name: str):
    m = r.get("metrics", {}).get(name)
    return m.get("value") if isinstance(m, dict) else m


def q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main(path: str) -> None:
    if os.path.isdir(path):
        path = os.path.join(path, "profile_export.jsonl")
    recs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("metadata", {}).get("benchmark_phase") == "profiling":
            recs.append(r)

    rows = []
    for r in recs:
        ttft = mv(r, "time_to_first_token")
        isl = mv(r, "input_sequence_length") or 0
        cached = mv(r, "usage_prompt_cache_read_tokens") or 0
        if ttft is None:
            continue
        rows.append({
            "ttft": ttft,
            "isl": isl,
            "cached": cached,
            "new": max(0, isl - cached),
            "tps": mv(r, "prefill_throughput_per_user") or 0.0,
            "turn": r.get("metadata", {}).get("turn_index"),
        })
    n = len(rows)
    if not n:
        print("no records")
        return

    # A rate this high is only reachable when a request has the machine to
    # itself, so treating it as the floor deliberately over-attributes to
    # queueing rather than under-attributing.
    peak_tps = q([x["tps"] for x in rows if x["tps"] > 0], 0.95) or 1.0
    for x in rows:
        x["prefill_floor"] = 1000.0 * x["new"] / peak_tps
        x["queue"] = max(0.0, x["ttft"] - x["prefill_floor"])

    total_pts = sum(W * 100 * s_ttft(x["ttft"]) for x in rows) / n
    print(f"n={n}   peak prefill (p95) = {peak_tps:,.0f} tok/s")
    print(f"s_ttft points earned = {total_pts:.2f} / {W*100:.0f}"
          f"   (lost {W*100-total_pts:.2f})\n")

    print(f"{'new tokens':>16}{'n':>6}{'%':>7}{'ttft p50':>10}{'ttft p90':>10}"
          f"{'>6s':>7}{'queue%':>8}{'pts lost':>10}")
    print("-" * 74)
    for lo, hi in BUCKETS:
        b = [x for x in rows if lo <= x["new"] < hi]
        if not b:
            continue
        tt = [x["ttft"] for x in b]
        lost = sum(W * 100 * (1 - s_ttft(x["ttft"])) for x in b) / n
        qfrac = st.mean([x["queue"] / x["ttft"] for x in b if x["ttft"] > 0])
        over = sum(1 for x in tt if x >= C_TTFT) / len(b)
        name = f"{lo//1000}k-{'inf' if hi > 10**8 else str(hi//1000)+'k'}"
        print(f"{name:>16}{len(b):>6}{100*len(b)/n:>6.1f}%{q(tt,.5):>10.0f}"
              f"{q(tt,.9):>10.0f}{100*over:>6.0f}%{100*qfrac:>7.0f}%{lost:>10.2f}")

    print()
    first = [x for x in rows if x["turn"] == 0]
    later = [x for x in rows if x["turn"] not in (0, None)]
    for label, g in (("turn 0 (cache-bust)", first), ("turn >0", later)):
        if not g:
            continue
        lost = sum(W * 100 * (1 - s_ttft(x["ttft"])) for x in g) / n
        print(f"{label:>22}: n={len(g):>4} ({100*len(g)/n:>4.1f}%)  "
              f"ttft p50={q([x['ttft'] for x in g],.5):>7.0f}  "
              f"hit={100*st.mean([x['cached']/max(1,x['isl']) for x in g]):>5.1f}%  "
              f"pts lost={lost:.2f}")

    q_lost = sum(W * 100 * (s_ttft(x["prefill_floor"]) - s_ttft(x["ttft"]))
                 for x in rows) / n
    print(f"\npoints recoverable by removing ALL queueing = {q_lost:.2f}"
          f"   (i.e. every request served at {peak_tps:,.0f} tok/s the instant it arrives)")
    hit = st.mean([x["cached"] / max(1, x["isl"]) for x in rows])
    print(f"mean per-request cache hit = {100*hit:.1f}%   "
          f"new tokens: mean={st.mean([x['new'] for x in rows]):,.0f} "
          f"p50={q([x['new'] for x in rows],.5):,.0f} "
          f"p90={q([x['new'] for x in rows],.9):,.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
