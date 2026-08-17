#!/bin/bash
# Build the kernel cache once, so no measured run ever pays for compilation again.
#
# Kernels here are templates over shape and dtype; each concrete combination is
# generated and compiled on first use, and FlashInfer's default cache directory
# lives inside the container. Every restart therefore threw the cache away and
# re-paid nvcc for everything -- 28.8 s for a single shape, and eight points of
# score spread over the first 600 s of a run, with first tokens as long as 16.5 s
# for requests that had 398 new tokens to process.
#
# This walks both KV dtypes through the context range the replay visits, writing
# into one mounted directory. fp8 and bf16 kernels are keyed separately and
# coexist there, so the same cache serves either configuration afterwards.
#
# Compilation is left unrestricted on purpose: nothing is being served, so there
# is no request for nvcc to starve, and all 22 cores should finish it as fast as
# possible. During a *measured* run the opposite holds -- pass MAX_JOBS=2 there.
#
# The last pass is the point of the exercise: it re-boots the first dtype and
# warms it again. If the cache works, its reported compile cost is near zero.
#
# Usage: nohup prime_cache.sh &
set -u
ROOT=${ROOT:-/srv/contest-workspace}
BIN=${BIN:-$(cd "$(dirname "$0")" && pwd)}
FI_CACHE=${FI_CACHE:-$ROOT/flashinfer_cache}
LOG=${LOG:-$BIN/prime_cache.log}
export FI_CACHE

{
  echo "=== priming kernel cache at $FI_CACHE  $(date -Is)"
  du -sh "$FI_CACHE" 2>/dev/null || echo "(cache does not exist yet)"

  for dtype in fp8 bf16; do
    echo
    echo "=== [$dtype] boot"
    bash "$BIN/serve_kv.sh" "$dtype" || { echo "SERVE_FAILED $dtype"; continue; }

    echo "=== [$dtype] warm every context length the replay will visit"
    python3 "$BIN/warmup.py" || echo "WARMUP_FAILED $dtype"

    echo "=== [$dtype] kernels compiled during that pass"
    docker logs llm-serve 2>&1 | grep -ciE "JIT compilation during inference" || true
    du -sh "$FI_CACHE" 2>/dev/null
    find "$FI_CACHE" -maxdepth 2 -type d | wc -l

    docker update --restart=no llm-serve >/dev/null 2>&1
    docker rm -f llm-serve >/dev/null 2>&1
    while docker ps -a --format '{{.Names}}' | grep -qx llm-serve; do sleep 2; done
  done

  echo
  echo "=== VERIFY: fresh container, cache already on disk"
  bash "$BIN/serve_kv.sh" fp8 || { echo "SERVE_FAILED verify"; exit 1; }
  python3 "$BIN/warmup.py" || echo "WARMUP_FAILED verify"
  echo "=== JIT during the verify pass (want 0)"
  docker logs llm-serve 2>&1 | grep -ciE "JIT compilation during inference" || true

  echo
  echo "=== cache now"
  du -sh "$FI_CACHE"
  echo "=== done $(date -Is)"
} > "$LOG" 2>&1
echo "PRIME_CACHE_DONE"
