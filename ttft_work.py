#!/usr/bin/env python3
"""Is a slow first token doing work, or waiting for company?

The two answers want opposite fixes, and the engine's own counters cannot tell
them apart: `request_queue_time` only covers the wait before a request is first
scheduled. Under chunked prefill a request can be running and still be starved
of tokens because another request is taking the step's budget, and that time
lands in `prefill_time` looking exactly like computation.

So the work has to be modelled and subtracted. Attention during prefill costs
`n_new x n_ctx` plus a causal triangle among the new tokens themselves, and both
factors swing by two orders of magnitude across this workload -- a turn that
reuses 99% of a 200k prompt does almost nothing, while a cache-busted first turn
at the same length does all of it. Fitting time against new tokens alone, as the
earlier tool does, leaves the context-length effect in the residual, where it is
easy to mistake for contention.

    ttft = b + a1*new + a2*(new*ctx + new^2/2) + a3*(requests in flight)

`a3` is the number the scheduler question turns on, and the honest way to read it
is the R-squared it adds over the work-only model: if the work terms already
explain the variance, a long first token is expensive, not queued, and no
scheduling policy shortens it.

Usage:  python3 ttft_work.py <artifact_dir> [more dirs...]
"""
from __future__ import annotations

import json
import os
import sys

PHASE = "profiling"


def metric(rec: dict, name: str):
    m = rec.get("metrics", {}).get(name)
    return m.get("value") if isinstance(m, dict) else m


def load(path: str, after: float = 0.0) -> list[dict]:
    """`after` drops the first N seconds of the run.

    Not cosmetic: a cold engine compiles kernels on the first request that needs
    each shape, and a single nvcc call costs tens of seconds. Those stalls have
    nothing to do with either work or contention, and they are large enough to
    swamp both -- fitting across them produced an R-squared of 0.03 and told us
    nothing. Anything asking what the steady state does has to start after the
    compiling stops.
    """
    if os.path.isdir(path):
        path = os.path.join(path, "profile_export.jsonl")
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        md = r.get("metadata", {})
        if md.get("benchmark_phase") != PHASE:
            continue
        ttft = metric(r, "time_to_first_token")
        isl = metric(r, "input_sequence_length") or 0
        cached = metric(r, "usage_prompt_cache_read_tokens") or 0
        if not ttft or "request_start_ns" not in md:
            continue
        new = max(isl - cached, 0)
        out.append({
            "ttft": float(ttft),
            "new": float(new),
            "ctx": float(cached),
            # Pairs of (query, key) the chunked prefill actually scores.
            "pairs": new * cached + new * new / 2.0,
            "start": md["request_start_ns"] / 1e6,
            "first": md["request_start_ns"] / 1e6 + float(ttft),
        })
    if out and after > 0:
        t0 = min(r["start"] for r in out)
        out = [r for r in out if r["start"] - t0 >= after * 1000.0]
    return out


def add_concurrency(rs: list[dict]) -> None:
    """How many other requests were mid-prefill when this one asked."""
    for r in rs:
        r["conc"] = sum(1 for o in rs
                        if o is not r and o["start"] <= r["start"] <= o["first"])


def solve(cols: list[list[float]], y: list[float]) -> list[float] | None:
    """Normal equations by elimination. The system is 4x4; numpy is not assumed."""
    n = len(cols[0])
    a = [[sum(c[i] * c[j] for c in cols) for j in range(n)]
         + [sum(c[i] * v for c, v in zip(cols, y))] for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda k: abs(a[k][i]))
        a[i], a[piv] = a[piv], a[i]
        if abs(a[i][i]) < 1e-16:
            return None
        for k in range(i + 1, n):
            f = a[k][i] / a[i][i]
            for j in range(i, n + 1):
                a[k][j] -= f * a[i][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def r2(cols, y, x) -> tuple[float, list[float]]:
    pred = [sum(c[i] * x[i] for i in range(len(x))) for c in cols]
    mean = sum(y) / len(y)
    sse = sum((v - p) ** 2 for v, p in zip(y, pred))
    sst = sum((v - mean) ** 2 for v in y)
    return (1 - sse / sst if sst else float("nan")), [v - p for v, p in zip(y, pred)]


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def report(path: str, after: float = 0.0) -> None:
    rs = load(path, after)
    name = path.rstrip("/").split("/")[-1]
    if len(rs) < 10:
        print(f"\n=== {name}: only {len(rs)} usable records, not fitting")
        return
    add_concurrency(rs)
    y = [r["ttft"] for r in rs]

    work = [[1.0, r["new"], r["pairs"]] for r in rs]
    full = [[1.0, r["new"], r["pairs"], float(r["conc"])] for r in rs]
    xw, xf = solve(work, y), solve(full, y)
    if xw is None or xf is None:
        print(f"\n=== {name}: singular fit")
        return
    r2w, _ = r2(work, y, xw)
    r2f, res = r2(full, y, xf)

    print(f"\n=== {name}   n={len(rs)}  ttft p50={pct(y,.5):.0f} p90={pct(y,.9):.0f} "
          f"p99={pct(y,.99):.0f} ms   in-flight p50={pct([r['conc'] for r in rs],.5):.0f} "
          f"p90={pct([r['conc'] for r in rs],.9):.0f}")
    print(f"  work only  : R2 = {r2w:.3f}")
    print(f"  + in flight: R2 = {r2f:.3f}   (adds {r2f - r2w:+.3f})")
    print(f"  ttft ms = {xf[0]:.0f}"
          f" + {xf[1]*1000:.1f} per 1k new tok"
          f" + {xf[2]*1e9:.2f} per 1e9 pairs"
          f" + {xf[3]:.0f} per request in flight")
    q50 = pct([r["conc"] for r in rs], .5)
    q90 = pct([r["conc"] for r in rs], .9)
    print(f"  company term: {xf[3]*q50:.0f} ms at median in-flight ({q50:.0f}),"
          f" {xf[3]*q90:.0f} ms at p90 ({q90:.0f})")
    print(f"  residual ms : p50={pct(res,.5):+.0f} p90={pct(res,.9):+.0f} "
          f"max={max(res):+.0f}")

    worst = sorted(zip(res, rs), key=lambda t: -t[0])[:5]
    print(f"  {'resid':>8}{'ttft':>8}{'new':>9}{'ctx':>9}{'inflt':>7}   slowest vs model")
    for d, r in worst:
        print(f"  {d:>+8.0f}{r['ttft']:>8.0f}{r['new']:>9.0f}{r['ctx']:>9.0f}{r['conc']:>7.0f}")

    verdict = ("contention is real -- the scheduler axis is open"
               if r2f - r2w > 0.05 and xf[3] * q90 > 200 else
               "work explains it -- no scheduling policy shortens these")
    print(f"  => {verdict}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--after")]
    after = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--after="):
            after = float(a.split("=", 1)[1])
    if after:
        print(f"(dropping the first {after:.0f}s of each run: kernel compilation)")
    for p in args or ["/srv/contest-workspace/bench/kv_B"]:
        report(p, after)
