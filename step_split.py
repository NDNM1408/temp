#!/usr/bin/env python3
"""What is the other third of a decode step?

The MoE microbenchmark accounts for 20 ms of a 33 ms step and the obvious
remainders -- paged attention over the resident contexts, the GDN state update,
the vocabulary projection -- add up to about 3 ms on paper. Roughly 10 ms per
step is unexplained, and since TPOT is the larger of the two score gaps that
gap is worth more than anything left on the TTFT side.

Guessing further is cheap and wrong; this reads it off a torch profiler trace
instead. The engine is started with a profiler directory already configured, so
a trace costs a load generator and two HTTP calls rather than a restart.

Kernels are bucketed by what they belong to rather than listed, because the
question is not which kernel is slowest but which *subsystem* owns the missing
time -- a hundred small elementwise launches and one big GEMM want completely
different fixes.

Usage:
    python3 step_split.py trace                 # drive load, capture, summarise
    python3 step_split.py report <trace.json>   # summarise an existing trace
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import sys
import threading
import time
import urllib.request

# The lab sandbox and the contest VM mount their profile directory in different
# places, so both are read from the environment with the lab's layout as default.
BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")
PROFILE_DIR = os.environ.get("PROFILE_DIR", "/lab/profiles")

# Ordered: the first pattern that matches a kernel name wins, so the specific
# expert-GEMM names have to be tested before the generic "gemm" catch-all.
BUCKETS = [
    ("moe", ("fused_moe", "moe_align", "grouped_gemm", "silu_and_mul",
             "moe_sum", "topk_softmax", "sgl_moe", "moe_wna16")),
    ("attention", ("flash", "paged", "attn", "reshape_and_cache", "rotary")),
    ("gdn/mamba", ("mamba", "ssm", "gdn", "causal_conv", "chunk_scan",
                   "chunk_state", "state_passing", "recurrent")),
    ("quant/cast", ("quant", "scaled_fp8", "to_fp8", "convert", "cast",
                    "dequant")),
    ("norm/elementwise", ("rms_norm", "layer_norm", "add_", "mul_", "elementwise",
                          "vectorized", "copy", "fill", "index", "cat", "silu")),
    ("gemm (dense)", ("gemm", "cutlass", "sm90", "cublas", "matmul", "nvjet")),
    ("sample/logits", ("softmax", "argmax", "sort", "top_k", "top_p", "gather",
                       "random", "multinomial", "logit")),
    ("comm/transfer", ("memcpy", "memset", "nccl", "dtoh", "htod")),
]


def post(path: str, body: dict | None = None) -> None:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, pats in BUCKETS:
        if any(p in low for p in pats):
            return label
    return "other"


def drive(concurrency: int, prompt_tokens: int, out_tokens: int) -> None:
    """Hold `concurrency` decodes in flight, which is what the step shape is."""
    # Digit junk tokenises at roughly one token per character group and never
    # hits the prefix cache across workers, so each stream really does prefill.
    body = lambda i: {
        "model": "Qwen3.5-122B-A10B-FP8",
        "messages": [{"role": "user",
                      "content": f"{i} " + " ".join(str(j) for j in range(prompt_tokens // 2))}],
        "max_tokens": out_tokens, "temperature": 0.0, "ignore_eos": True,
    }
    errs: list[str] = []

    def one(i: int) -> None:
        try:
            post("/v1/chat/completions", body(i))
        except Exception as exc:  # a dead stream would otherwise look like idle
            errs.append(repr(exc))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        print(f"  {len(errs)} request(s) failed: {errs[0]}")


def newest_trace() -> str | None:
    files = glob.glob(os.path.join(PROFILE_DIR, "**", "*.json*"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def report(path: str) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        trace = json.load(fh)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    # Device-side duration is what a step actually costs; host-side ops overlap
    # with it and would double count.
    by_bucket: dict[str, float] = collections.defaultdict(float)
    by_kernel: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    total = 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy",
                                                      "gpu_memset"):
            continue
        dur = e.get("dur", 0) / 1000.0
        name = e.get("name", "?")
        by_bucket[bucket_of(name)] += dur
        by_kernel[name][0] += dur
        by_kernel[name][1] += 1
        total += dur

    if not total:
        print("no device kernels in trace")
        return
    print(f"\ndevice time = {total:.1f} ms across {len(by_kernel)} distinct kernels\n")
    print(f"{'bucket':>18}{'ms':>10}{'%':>8}")
    print("-" * 36)
    for label, ms in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        print(f"{label:>18}{ms:>10.1f}{100*ms/total:>7.1f}%")

    print(f"\n{'top kernels':>18}{'ms':>10}{'%':>8}{'calls':>8}   bucket")
    print("-" * 62)
    for name, (ms, n) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:15]:
        short = name if len(name) <= 46 else name[:43] + "..."
        print(f"{ms:>28.1f}{100*ms/total:>7.1f}%{n:>8}   {bucket_of(name)}  {short}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "trace"
    if mode == "report":
        report(sys.argv[2])
        return

    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    # Both of these decide what the trace is a trace *of*. Attention cost scales
    # with resident context and MoE cost with how many experts the batch reaches,
    # so a capture at a short context or a thin batch produces a believable
    # breakdown of the wrong machine. The defaults are the measured operating
    # point: 15 concurrent decodes over ~65k-token contexts.
    ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 65000
    before = set(glob.glob(os.path.join(PROFILE_DIR, "**", "*.json*"), recursive=True))

    # The warm-up pass pays the prefill; the capture pass reuses the same prompts
    # so they hit the prefix cache and the trace is decode, not prefill.
    print(f"warming {conc} streams at ~{ctx} ctx to steady-state decode...")
    drive(conc, ctx, 64)

    print("capturing...")
    post("/start_profile")
    drive(conc, ctx, 96)
    post("/stop_profile")

    for _ in range(60):  # the writer flushes well after the HTTP call returns
        new = set(glob.glob(os.path.join(PROFILE_DIR, "**", "*.json*"),
                            recursive=True)) - before
        if new:
            path = max(new, key=os.path.getmtime)
            size = os.path.getsize(path)
            time.sleep(3)
            if os.path.getsize(path) == size:
                print(f"trace: {path} ({size/1e6:.0f} MB)")
                report(path)
                return
        time.sleep(5)
    print(f"no new trace appeared under {PROFILE_DIR}")


if __name__ == "__main__":
    main()
