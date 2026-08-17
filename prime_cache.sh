#!/bin/bash
# Build the kernel cache once, so no measured run ever pays for compilation again.
#
# FlashInfer's cache directory defaults to a path inside the container, so every
# restart threw the compiled kernels away and re-paid nvcc for all of them: 28.8 s
# for one shape, and eight points of score spread across the first 600 s of a run.
# serve_kv.sh now mounts that directory; this fills it.
#
# Four configurations are walked, because a kernel's identity includes the KV
# dtype and the token budget changes the chunk sizes the MoE and Triton kernels
# specialise on. They share one cache directory -- entries are keyed, not
# overwritten -- so each pass after the first is mostly cache hits and runs fast.
# Production's configuration goes first, so an interrupted run still leaves the
# one that matters most complete.
#
# Compilation is deliberately unrestricted: nothing is being served, so nvcc has
# no request to starve and should use every core. During a *measured* run the
# opposite holds, which is what MAX_JOBS=2 is for there.
#
# The last pass is the whole point. It boots a fresh container on production's
# configuration and counts the compilations the engine reports during inference.
# That count is the claim being tested, and zero is the only passing answer.
#
# Usage: nohup prime_cache.sh &
set -u
ROOT=${ROOT:-/srv/contest-workspace}
BIN=${BIN:-$(cd "$(dirname "$0")" && pwd)}
FI_CACHE=${FI_CACHE:-$ROOT/flashinfer_cache}
LOG=${LOG:-$BIN/prime_cache.log}
export FI_CACHE

stop_server() {
  docker update --restart=no llm-serve >/dev/null 2>&1
  docker rm -f llm-serve >/dev/null 2>&1
  while docker ps -a --format '{{.Names}}' | grep -qx llm-serve; do sleep 2; done
}

# dtype:budget, production first
PASSES=${PASSES:-"fp8:16384 fp8:8192 bf16:16384 bf16:8192"}

{
  echo "=== priming kernel cache at $FI_CACHE   $(date -Is)"
  du -sh "$FI_CACHE" 2>/dev/null || echo "(cache does not exist yet)"

  for pass in $PASSES; do
    dtype=${pass%%:*}
    mnbt=${pass##*:}
    echo
    echo "############ pass dtype=$dtype mnbt=$mnbt   $(date -Is)"
    stop_server
    MNBT=$mnbt bash "$BIN/serve_kv.sh" "$dtype" || { echo "SERVE_FAILED"; continue; }

    python3 "$BIN/warmup.py" || echo "WARMUP_FAILED"

    echo "--- compilations the engine reported during this pass:"
    docker logs llm-serve 2>&1 | grep -icE "JIT compilation during inference" || true
    docker logs llm-serve 2>&1 | grep -oE "JIT compilation during inference: [a-z_]+" \
      | sort -u | sed 's/^/    /'
    echo "--- cache size now: $(du -sh "$FI_CACHE" 2>/dev/null | cut -f1)"
  done

  echo
  echo "############ VERIFY on a fresh container, production configuration"
  stop_server
  MNBT=16384 bash "$BIN/serve_kv.sh" fp8 || { echo "SERVE_FAILED verify"; exit 1; }
  python3 "$BIN/warmup.py" || echo "WARMUP_FAILED verify"
  echo "--- JIT compilations during the verify pass (0 is the passing answer):"
  docker logs llm-serve 2>&1 | grep -icE "JIT compilation during inference" || true
  docker logs llm-serve 2>&1 | grep -oE "JIT compilation during inference: [a-z_]+" \
    | sort -u | sed 's/^/    /'
  echo "--- boot log: does it still say the GDN kernel is being compiled?"
  docker logs llm-serve 2>&1 | grep -iE "JIT-compiled|first run may take" | head -3

  echo
  echo "=== final cache: $(du -sh "$FI_CACHE" | cut -f1)   $(date -Is)"
  echo "=== done"
} > "$LOG" 2>&1
echo "PRIME_CACHE_DONE"
