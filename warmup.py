#!/usr/bin/env python3
"""Compile every kernel shape the replay can reach, before anything is measured.

Kernels here are templates over shape and dtype, generated and compiled on first
use. FlashInfer goes through nvcc -- 28.8 s for a single shape on this box --
and vLLM's Triton kernels through LLVM, faster but numerous. Startup only
exercises what its dummy inputs happen to touch; everything else arrives with
real traffic and blocks the request that needed it, mid-forward, with nothing to
return until the compiler finishes. Measured consequence of skipping this: a cold
run scored eight points below a warm one, 15% of requests past the six-second
mark against 0% in the same run's final third, and a 16.5 s first token for a
request whose prefill was 398 tokens.

Covering "a few prompt sizes" is not enough, because shape is not one axis:

  context length     autotune picks per shape, and attention cost scales with it
  partial chunks     the last chunk of a prefill is whatever is left over, so
                     token counts that are not multiples of the budget matter
  decode batch       kernels key on batch size; sending one request at a time
                     only ever compiles the batch-of-one variant
  mixed steps        a step holding a prefill chunk *and* running decodes is a
                     third shape again, and only concurrency produces it
  kv dtype           fp8 and bf16 are separate kernels (the caller sweeps both)

So this drives all five. The point is not to measure anything -- it is to make
the compiler run now, while nothing is being timed.

    python3 warmup.py                 # full sweep
    python3 warmup.py --quick         # context sweep only, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

MODEL = "Qwen3.5-122B-A10B-FP8"
# How many tokens a digit group costs is not a constant -- the numbers grow from
# one digit to five -- and guessing it wrong is how earlier versions of this file
# both overshot the model's context limit and, after over-correcting, topped out
# at 139k when the sweep was supposed to reach 200k. So it is measured against
# the server's own token count once, at startup, and used from there.
TOKENS_PER_GROUP = 4.0

# Spread across the range the replay visits, and deliberately not all multiples
# of the 8192/16384 token budget: the remainder decides the last chunk's shape.
CONTEXTS = [4000, 8000, 9000, 13000, 16000, 20000, 32000, 40000,
            64000, 72000, 96000, 128000, 160000, 200000]

# Decode batch sizes to force into existence. The live batch measured 0-2 at the
# concurrency this serves, but the seat limit is 32 and a burst reaches higher,
# so the cheap end of the ladder is covered generously.
BATCHES = [1, 2, 3, 4, 6, 8, 12]

_lock = threading.Lock()
_fail: list[str] = []


def junk_groups(groups: int, salt: int) -> str:
    return f"{salt} " + " ".join(str(i % 100000) for i in range(max(1, groups)))


def junk(target_tokens: int, salt: int) -> str:
    return junk_groups(int(target_tokens / TOKENS_PER_GROUP), salt)


def calibrate(base: str) -> None:
    """Ask the server what a known number of groups actually tokenises to."""
    global TOKENS_PER_GROUP
    probe = 4000
    _, tok = ask(base, junk_groups(probe, 99), max_tokens=1)
    if tok > 0:
        TOKENS_PER_GROUP = tok / probe
    print(f"calibration: {TOKENS_PER_GROUP:.2f} tokens per digit group "
          f"({probe} groups -> {tok} tokens)")


def ask(base: str, prompt: str, max_tokens: int = 8) -> tuple[float, int]:
    """Returns (wall ms, prompt tokens the server counted). Never raises."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        with _lock:
            _fail.append(f"HTTP {exc.code}")
        return -1.0, 0
    except Exception as exc:
        with _lock:
            _fail.append(type(exc).__name__)
        return -1.0, 0
    return (time.monotonic() - t0) * 1000, payload.get("usage", {}).get("prompt_tokens", 0)


def phase_contexts(base: str) -> float:
    """One prompt per context length, twice each: the gap is compile time."""
    print(f"\n[1/4] prefill shapes across the context range")
    print(f"{'target':>8}{'actual':>9}{'first ms':>10}{'repeat ms':>11}{'compile ms':>12}")
    print("-" * 50)
    total = 0.0
    for n in CONTEXTS:
        first, tok = ask(base, junk(n, 1))
        if first < 0:
            print(f"{n:>8}{'':>9}  skipped ({_fail[-1]})")
            continue
        repeat, _ = ask(base, junk(n, 2))
        gap = max(first - (repeat if repeat > 0 else first), 0.0)
        total += gap
        print(f"{n:>8}{tok:>9}{first:>10.0f}{repeat:>11.0f}{gap:>12.0f}")
    return total


def phase_batches(base: str) -> None:
    """Concurrent decodes, so batch-of-N kernels exist before traffic needs them.

    Prompts are short and generations long: the point is to have N sequences in
    the decode phase simultaneously, not to prefill anything.
    """
    print(f"\n[2/4] decode batch sizes")
    for n in BATCHES:
        threads = [threading.Thread(target=ask, args=(base, junk(2000, 1000 + i), 150))
                   for i in range(n)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(f"  batch {n:>3}: {1000*(time.monotonic()-t0):>8.0f} ms")


def phase_mixed(base: str) -> None:
    """Steps that hold a prefill chunk and running decodes at the same time.

    A long generation is started first, then prefills are fired while it is still
    producing tokens. That combination never occurs when requests are sent one at
    a time, and it is most of what the replay actually does.
    """
    print(f"\n[3/4] mixed prefill + decode steps")
    for ctx in (16000, 72000, 200000):
        keep = [threading.Thread(target=ask, args=(base, junk(3000, 2000 + i), 400))
                for i in range(3)]
        for t in keep:
            t.start()
        time.sleep(2)                      # let them reach the decode phase
        t0 = time.monotonic()
        ask(base, junk(ctx, 3000))         # prefill lands in a step with decodes
        print(f"  prefill {ctx:>7} alongside 3 decodes: {1000*(time.monotonic()-t0):>8.0f} ms")
        for t in keep:
            t.join()


def phase_confirm(base: str) -> float:
    """Re-run the context sweep. Compile time should now be ~0 everywhere."""
    print(f"\n[4/4] confirm: same shapes again, nothing should compile")
    worst = 0.0
    for n in CONTEXTS:
        a, _ = ask(base, junk(n, 7))
        b, _ = ask(base, junk(n, 8))
        if a > 0 and b > 0:
            worst = max(worst, abs(a - b))
    print(f"  largest first-vs-repeat gap left: {worst:.0f} ms")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--quick", action="store_true", help="context sweep only")
    args = ap.parse_args()

    t0 = time.monotonic()
    calibrate(args.base)
    compiled = phase_contexts(args.base)
    if not args.quick:
        phase_batches(args.base)
        phase_mixed(args.base)
        residual = phase_confirm(args.base)
    else:
        residual = float("nan")

    print("\n" + "=" * 50)
    print(f"compile time observed in phase 1 : {compiled/1000:>7.1f} s")
    if residual == residual:
        print(f"gap remaining after everything   : {residual:>7.0f} ms")
    print(f"wall clock                       : {(time.monotonic()-t0)/60:>7.1f} min")
    if _fail:
        print(f"requests that failed: {len(_fail)} ({sorted(set(_fail))})")
    print("\nA large phase-1 number and a small phase-4 number is the intended shape:")
    print("the compiler ran here instead of during something that was being timed.")
    print("The cache only survives restarts if FLASHINFER_CACHE_DIR points at a mount.")


if __name__ == "__main__":
    main()
