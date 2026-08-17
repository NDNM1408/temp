#!/usr/bin/env python3
"""Pay the engine's kernel compilation before a timed run, and price it.

Attention, linear-attention and MoE kernels are templates over shape and dtype,
not fixed binaries. Each concrete combination is generated and compiled on first
use -- FlashInfer through nvcc, which measured 28.8 s for one shape here, Triton
through LLVM, which is faster but happens for many kernels. Startup only covers
what its dummy inputs happen to exercise; the rest arrives with real traffic and
blocks the request that needed it, mid-forward, with nothing to return until the
compiler finishes.

Measured cost of skipping this: a cold run scored 8 points below a warm one, with
15% of requests past the 6 s mark against 0% in the same run's last third, and
the eight worst first tokens all inside the first 288 s -- one of them 16.5 s for
a request with 398 new tokens, which is no work at all.

Content is irrelevant: a compile keys on shapes and dtypes. Digit junk is used
because it never matches the prefix cache, so it cannot leave blocks behind that
distort a later hit rate. Length is what matters, so the sweep covers the range
the run will actually see.

Two earlier mistakes are fixed here. Sizes are in *tokens*, converted using the
measured ~8 tokens per digit group -- assuming one token per group asked for a
560k-token prompt, which the server rejected and which killed the whole warm-up
after its first size. And a failure on one size no longer aborts the rest.

    python3 warmup.py                     # sweep, then report compile cost
    python3 warmup.py --sizes 8000 200000
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

MODEL = "Qwen3.5-122B-A10B-FP8"
# Digit groups tokenise at roughly eight tokens per group for this tokenizer.
TOKENS_PER_GROUP = 8.0


def junk(target_tokens: int, salt: int) -> str:
    groups = max(1, int(target_tokens / TOKENS_PER_GROUP))
    return f"{salt} " + " ".join(str(i % 100000) for i in range(groups))


def ask(base: str, prompt: str, max_tokens: int = 16) -> tuple[float, int]:
    """Returns (wall ms, prompt tokens the server counted)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        payload = json.loads(resp.read())
    ms = (time.monotonic() - t0) * 1000
    return ms, payload.get("usage", {}).get("prompt_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    # Spread across the context range the replay visits: short turns, the median
    # around 70k, and the 200k ceiling the client cap allows.
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[8000, 32000, 72000, 128000, 200000])
    args = ap.parse_args()

    print(f"{'target':>8}{'actual':>9}{'first ms':>10}{'repeat ms':>11}{'compile ms':>12}")
    print("-" * 50)
    total = 0.0
    for n in args.sizes:
        try:
            first, tok = ask(args.base, junk(n, 1))
            repeat, _ = ask(args.base, junk(n, 2))   # same shape, new content
        except urllib.error.HTTPError as exc:
            print(f"{n:>8}{'':>9}  HTTP {exc.code}: {exc.read()[:60].decode(errors='replace')}")
            continue
        except Exception as exc:                      # keep going; a missed size
            print(f"{n:>8}{'':>9}  {type(exc).__name__}: {str(exc)[:50]}")
            continue                                  # is better than no warm-up
        gap = max(first - repeat, 0.0)
        total += gap
        print(f"{n:>8}{tok:>9}{first:>10.0f}{repeat:>11.0f}{gap:>12.0f}")

    print("-" * 50)
    print(f"one-time compile cost paid here: ~{total/1000:.1f} s")
    print("\nThis is only the part paid for the shapes above. The run will still meet")
    print("combinations this sweep did not, so the durable fix is a persistent")
    print("FLASHINFER_CACHE_DIR -- warming a fresh container every time re-pays a")
    print("cost that was already paid on an earlier boot.")


if __name__ == "__main__":
    main()
