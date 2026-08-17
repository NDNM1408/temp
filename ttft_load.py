#!/usr/bin/env python3
"""Split a loaded TTFT into the part that scales with work and the part that
scales with company.

The idle probe priced two constants every request pays no matter what: about
110 ms of tokenising in the frontend and about 130 ms of engine floor. At the
median that leaves roughly two thirds of the observed TTFT unexplained, and
"contention" is not one thing. Two candidates behave differently and want
different fixes:

  work in flight   the engine is mid-step on somebody else's tokens, so a new
                   prefill waits for the step boundary and then shares the
                   batch. Scales with how many tokens the machine is chewing,
                   and shortens if the step shortens.
  company at the   the frontend serialises: one process tokenising a 66k-token
  door             prompt while thirty-one others queue behind it. Scales with
                   the number of simultaneous arrivals and is blind to size.

Both look like "TTFT got worse under load" in the aggregate. They separate in
the per-request records, which already carry arrival and completion stamps and
the cached-token count, so no rerun is needed:

    python3 ttft_load.py <records.jsonl> [...]

Reported per file: TTFT against tokens actually computed, against arrivals in
flight, and a joint fit whose coefficients are the per-token and per-neighbour
prices. A large per-neighbour price at small token counts is the frontend.
"""
from __future__ import annotations

import bisect
import json
import sys


def load(path: str) -> list[dict]:
    """Scored requests only: warmup runs against a cold cache and a cold engine."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            md, mt = r.get("metadata", {}), r.get("metrics", {})
            if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled"):
                continue
            ttft = mt.get("time_to_first_token", {}).get("value")
            isl = mt.get("input_sequence_length", {}).get("value")
            if ttft is None or not isl:
                continue
            cached = mt.get("usage_prompt_cache_read_tokens", {}).get("value", 0) or 0
            out.append({
                "ttft": ttft,
                "new": max(isl - cached, 0),
                "isl": isl,
                "cached": cached,
                "start": md["request_start_ns"] / 1e6,
                "end": md["request_end_ns"] / 1e6,
                "send": mt.get("http_req_sending", {}).get("value", 0.0),
                "wait": mt.get("http_req_waiting", {}).get("value", 0.0),
                "reused": mt.get("http_req_connection_reused", {}).get("value", 0.0),
            })
    out.sort(key=lambda r: r["start"])
    return out


def with_concurrency(rs: list[dict]) -> None:
    """How many other requests were open when each one arrived."""
    starts = [r["start"] for r in rs]
    ends = sorted(r["end"] for r in rs)
    for r in rs:
        opened = bisect.bisect_right(starts, r["start"])
        closed = bisect.bisect_right(ends, r["start"])
        r["conc"] = opened - closed - 1
        # Tokens the engine still owed at this instant, as a stand-in for the
        # depth of the batch a newcomer has to share.
        r["load"] = sum(o["new"] for o in rs
                        if o["start"] <= r["start"] < o["end"] and o is not r)


def s_ttft(ms: float, f: float = 200.0, c: float = 6000.0) -> float:
    x = max(min((c - ms) / (c - f), 1.0), 0.0)
    return x * x


def p(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)]


def bucket(rs: list[dict], key: str, edges: list[float], label: str,
           n_all: int = 0) -> None:
    """Points are always charged against the whole run, so a slice of a slice
    stays comparable with the top-level table."""
    print(f"\n  {label:>14}{'n':>6}{'ttft p50':>10}{'p90':>8}"
          f"{'new p50':>10}{'conc p50':>10}{'pts lost':>10}")
    n_all = n_all or len(rs)
    for lo, hi in zip(edges, edges[1:] + [float("inf")]):
        g = [r for r in rs if lo <= r[key] < hi]
        if not g:
            continue
        lost = sum(1 - s_ttft(r["ttft"]) for r in g) / n_all * 50
        name = f"{lo:g}-{hi:g}" if hi != float("inf") else f"{lo:g}+"
        print(f"  {name:>14}{len(g):>6}{p([r['ttft'] for r in g], .5):>10.0f}"
              f"{p([r['ttft'] for r in g], .9):>8.0f}"
              f"{p([r['new'] for r in g], .5):>10.0f}"
              f"{p([r['conc'] for r in g], .5):>10.0f}{lost:>10.2f}")


def fit(rs: list[dict]) -> None:
    """Least squares of ttft on (1, new tokens, arrivals in flight).

    Solved with a plain 3x3 normal-equation elimination -- numpy is not always
    present and the system is tiny.
    """
    cols = [[1.0, r["new"], float(r["conc"])] for r in rs]
    y = [r["ttft"] for r in rs]
    n = 3
    a = [[sum(c[i] * c[j] for c in cols) for j in range(n)] + [sum(c[i] * v for c, v in zip(cols, y))]
         for i in range(n)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[pivot] = a[pivot], a[i]
        if abs(a[i][i]) < 1e-12:
            print("  fit: singular")
            return
        for k in range(i + 1, n):
            f = a[k][i] / a[i][i]
            for j in range(i, n + 1):
                a[k][j] -= f * a[i][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    mean = sum(y) / len(y)
    sse = sum((r["ttft"] - (x[0] + x[1] * r["new"] + x[2] * r["conc"])) ** 2 for r in rs)
    sst = sum((v - mean) ** 2 for v in y)
    print(f"\n  ttft ms = {x[0]:.0f} + {x[1]*1000:.2f} per 1k new tokens"
          f" + {x[2]:.1f} per concurrent arrival     R2={1-sse/sst:.3f}")
    q = p([r["conc"] for r in rs], .5)
    print(f"  at the median arrival ({q:.0f} in flight) the company term is"
          f" {x[2]*q:.0f} ms of a {p(y, .5):.0f} ms median")


def main() -> None:
    for path in sys.argv[1:]:
        rs = load(path)
        if not rs:
            print(f"\n=== {path.split('/')[-1]}: no scored records")
            continue
        with_concurrency(rs)
        ttfts = [r["ttft"] for r in rs]
        reuse = sum(r["reused"] for r in rs) / len(rs)
        print(f"\n=== {path.split('/')[-1]}  n={len(rs)}"
              f"  ttft p50={p(ttfts,.5):.0f} p90={p(ttfts,.9):.0f}"
              f"  s_ttft={sum(s_ttft(t) for t in ttfts)/len(rs):.4f}"
              f"  conn reuse={100*reuse:.0f}%")
        bucket(rs, "new", [0, 1000, 4000, 16000, 64000], "new tokens")
        bucket(rs, "conc", [0, 4, 8, 16, 24], "in flight")
        # The frontend story predicts that even requests with nothing to compute
        # slow down as arrivals pile up; the engine story predicts they do not.
        cheap = [r for r in rs if r["new"] < 1000]
        if cheap:
            print(f"\n  nothing to prefill (<1k new tokens), n={len(cheap)}:")
            bucket(cheap, "conc", [0, 4, 8, 16, 24], "in flight", len(rs))
            # A hit that lives in host memory is not free: its pages have to
            # cross PCIe before the first forward can run, and that cost tracks
            # the size of the hit, not the size of the work.
            print(f"\n  ...same requests, by how much cache they have to haul in"
                  f" (~12 KB/token):")
            bucket(cheap, "cached", [0, 8000, 32000, 64000], "cached tok", len(rs))
        fit(rs)


if __name__ == "__main__":
    main()
