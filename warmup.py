#!/usr/bin/env python3
"""Pay the engine's one-time compile costs before a timed run, and price them.

Two things in this stack compile at first use rather than at boot: the FlashInfer
linear-attention prefill kernel says so in the boot log, and FlashInfer's
autotuner picks a configuration the first time it sees a shape. Whatever that
costs lands on the first few requests of whatever runs next.

Content does not matter -- a JIT keys on shapes and dtypes -- so the prompts are
digit junk. Junk is better than real text here: it tokenises predictably and
never matches the prefix cache, so it cannot leave useful-looking blocks behind
that later distort a cache hit rate.

Length does matter. The autotuner specialises per shape, so a warm-up at 10k
says nothing about the kernel that will run at 200k.

The same pair of requests also prices the compile: two prompts of identical shape
and different content, back to back. The second cannot hit the cache, so the gap
between them is compile time and nothing else.

    python3 warmup.py                    # warm, then report the compile cost
    python3 warmup.py --sizes 8192       # one shape
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

MODEL = "Qwen3.5-122B-A10B-FP8"


def junk(tokens: int, salt: int) -> str:
    """Digit groups tokenise at roughly one token per group."""
    return f"{salt} " + " ".join(str(i % 100000) for i in range(tokens))


def ask(base: str, prompt: str, max_tokens: int = 16) -> float:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        resp.read()
    return (time.monotonic() - t0) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--sizes", type=int, nargs="*", default=[10000, 70000, 200000],
                   help="approximate prompt tokens; cover the range the run will see")
    args = ap.parse_args()

    print(f"{'tokens':>8} {'first ms':>10} {'repeat ms':>10} {'compile ms':>11}")
    print("-" * 43)
    total = 0.0
    for n in args.sizes:
        first = ask(args.base, junk(n, 1))
        repeat = ask(args.base, junk(n, 2))   # same shape, different content
        gap = first - repeat
        total += max(gap, 0.0)
        print(f"{n:>8} {first:>10.0f} {repeat:>10.0f} {gap:>11.0f}")

    print("-" * 43)
    print(f"one-time compile cost paid: ~{total:.0f} ms")
    print("\nIf that total is small, warming up is hygiene and nothing more: it can only")
    print("ever affect the first handful of requests. If it is seconds, warm up before")
    print("every timed run, and do not restart the container between warming and running.")


if __name__ == "__main__":
    main()
