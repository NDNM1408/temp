#!/usr/bin/env python3
"""How fast is prefill attention at this model's shape, and what is holding it back?

Prefill is 78% of time-to-first-token here, and full attention is 74% of prefill,
so the attention kernel alone decides more than half the latency budget. The
end-to-end number says it runs at roughly a fifth of the card's peak, but that
number is contaminated: it includes MoE, the linear-attention layers, paging and
scheduling. This times the attention call on its own.

Three things are being separated, and each one changes what to do next:

  head_dim      This checkpoint uses 256, which is unusual -- most attention
                kernels are written and tuned for 64 or 128. Running the same
                kernel at 128 with twice the heads keeps the FLOP count identical
                and isolates how much the wide head costs by itself. If the two
                land in the same place, the head width is not the problem and
                swapping backends will not help.

  kv dtype      An fp8 KV cache is normally a straight win, but on SM90 the fp8
                path carries a two-level accumulator added to protect long-context
                accuracy, and at head_dim 192/256 it spills registers badly enough
                to lose more than fp8 arithmetic gains. Whether that applies to
                this backend on this card has never been measured here, and it is
                a single flag either way.

  backend       Different kernels handle the same shape differently.

The metric is the slope of time against how much context is already cached, at a
fixed chunk size. That is the quantity that matters: chunked prefill runs a fixed
number of query tokens against a growing prefix, so this slope is what integrates
into the quadratic growth of TTFT with prompt length. A single measurement at one
context length hides it.

Runs against the GPU directly -- no server, no load generator.

    python3 attn_probe.py                          # every arm it can build
    python3 attn_probe.py --backends flashinfer    # one backend
    python3 attn_probe.py --kv-lens 8192 73728     # fewer points, faster
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

# The served shape. head_dim and the 32:2 head ratio come from the checkpoint;
# q_len is the scheduler's token budget, which is what one prefill step actually
# submits regardless of how long the prompt is.
HIDDEN_HEADS, KV_HEADS, HEAD_DIM = 32, 2, 256
Q_LEN = 8192
LAYERS = 12  # full-attention layers; the other 36 are linear-attention

# Dense peaks, no sparsity. Quoting the sparsity number would halve every MFU
# below for the same measurement, so the choice is stated rather than implied.
PEAK_TFLOPS = {"fp8": 1979.0, "bf16": 990.0}

FP8 = torch.float8_e4m3fn


def flops(q_len: int, kv_prev: int, head_dim: int, heads: int) -> float:
    """QK^T and AV, two operations each, over the pairs a chunk actually scores.

    The chunk sees every cached token, plus a causal triangle among its own.
    """
    pairs = q_len * kv_prev + q_len * q_len / 2
    return 4.0 * pairs * head_dim * heads


def build(kv_prev: int, head_dim: int, heads: int, kv_dtype: str):
    """Query in bf16, KV in the dtype under test -- the layout the engine uses."""
    kv_len = kv_prev + Q_LEN
    kv_heads = max(1, heads // (HIDDEN_HEADS // KV_HEADS))
    q = torch.randn(Q_LEN, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_len, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    if kv_dtype == "fp8":
        k, v = k.to(FP8), v.to(FP8)
    return q, k, v


# --- backend adapters -------------------------------------------------------
# Each returns a zero-argument callable, or None with a reason. Import paths and
# signatures move between releases, so every one is attempted and the failure is
# reported rather than raised -- "this arm cannot be built" is itself a result.

def adapter_flashinfer(q, k, v):
    import flashinfer
    fn = getattr(flashinfer, "single_prefill_with_kv_cache", None)
    if fn is None:
        fn = getattr(flashinfer.prefill, "single_prefill_with_kv_cache")
    return lambda: fn(q, k, v, causal=True, kv_layout="NHD")


def adapter_flash_attn(q, k, v):
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError:
        from flash_attn import flash_attn_varlen_func
    cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda")
    return lambda: flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=q.shape[0], max_seqlen_k=k.shape[0], causal=True)


def adapter_triton(q, k, v):
    from vllm.attention.ops.triton_flash_attention import triton_attention
    cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    return lambda: triton_attention(q, k, v, out, cu_q, cu_k,
                                    q.shape[0], k.shape[0], True, 1.0 / (HEAD_DIM ** 0.5))


ADAPTERS = {
    "flashinfer": adapter_flashinfer,
    "flash_attn": adapter_flash_attn,
    "triton": adapter_triton,
}


def time_call(run, iters: int) -> float:
    """Median ms. Median, not mean: a stray context switch should not set the number."""
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def slope(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least squares ms = a·kv_prev + b. `a` is the term that integrates into TTFT."""
    n = len(points)
    if n < 2:
        return float("nan"), float("nan")
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan"), float("nan")
    a = (n * sxy - sx * sy) / denom
    return a, (sy - a * sx) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=list(ADAPTERS))
    ap.add_argument("--kv-lens", type=int, nargs="*",
                    default=[8192, 32768, 73728, 131072, 204800],
                    help="tokens already cached when the chunk arrives")
    ap.add_argument("--head-dims", type=int, nargs="*", default=[256, 128])
    ap.add_argument("--kv-dtypes", nargs="*", default=["fp8", "bf16"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="/tmp/attn_probe.json")
    args = ap.parse_args()

    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"shape       : q_len {Q_LEN}, {HIDDEN_HEADS} q / {KV_HEADS} kv heads, "
          f"head_dim {HEAD_DIM}, {LAYERS} full-attention layers")
    print(f"peaks       : fp8 {PEAK_TFLOPS['fp8']:.0f} TFLOPS, "
          f"bf16 {PEAK_TFLOPS['bf16']:.0f} TFLOPS (dense, no sparsity)\n")

    rows, series = [], {}
    for backend in args.backends:
        for head_dim in args.head_dims:
            # Keep total FLOPs identical when the head narrows, so the only thing
            # that changed is the shape the kernel sees.
            heads = HIDDEN_HEADS * (HEAD_DIM // head_dim)
            for kv_dtype in args.kv_dtypes:
                tag = f"{backend:<11} hdim {head_dim:>3} {kv_dtype:>4}"
                pts = []
                for kv_prev in args.kv_lens:
                    q, k, v = build(kv_prev, head_dim, heads, kv_dtype)
                    try:
                        run = ADAPTERS[backend](q, k, v)
                        ms = time_call(run, args.iters)
                    except Exception as exc:
                        msg = str(exc).split("\n")[0][:70]
                        print(f"{tag}  kv {kv_prev:>7}  UNSUPPORTED: {msg}")
                        del q, k, v
                        torch.cuda.empty_cache()
                        break
                    f = flops(Q_LEN, kv_prev, head_dim, heads)
                    tfs = f / (ms / 1000) / 1e12
                    mfu = 100 * tfs / PEAK_TFLOPS[kv_dtype]
                    print(f"{tag}  kv {kv_prev:>7}  {ms:8.3f} ms/layer  "
                          f"{ms * LAYERS:8.1f} ms/step  {tfs:7.1f} TFLOPS  {mfu:5.1f}% MFU")
                    rows.append(dict(backend=backend, head_dim=head_dim,
                                     kv_dtype=kv_dtype, kv_prev=kv_prev,
                                     ms=ms, tflops=tfs, mfu=mfu))
                    pts.append((kv_prev, ms))
                    del q, k, v, run
                    torch.cuda.empty_cache()
                if len(pts) >= 2:
                    a, b = slope(pts)
                    series[tag.strip()] = a
                    print(f"{tag}  -> slope {a * 1e6:8.3f} us per 1k cached tokens, "
                          f"intercept {b:.3f} ms\n")

    if series:
        print("=" * 78)
        print("SLOPE RANKING — us per 1k cached tokens, per layer (lower is better).")
        print("This is the term that integrates into TTFT as prompts get longer;")
        print("compare fp8 against bf16 at hdim 256 to test the accumulator regression,")
        print("and hdim 256 against hdim 128 to test whether the wide head is the cost.")
        print("=" * 78)
        best = min(series.values())
        for tag, a in sorted(series.items(), key=lambda kv: kv[1]):
            print(f"  {tag:<28} {a * 1e6:8.3f}   {a / best:5.2f}x vs best")

    with open(args.out, "w") as fh:
        json.dump(dict(rows=rows, slopes=series), fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
