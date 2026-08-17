#!/bin/bash
# Close the two coverage gaps the first priming pass left, then verify properly.
#
# The first pass warmed with `temperature: 0` and no `ignore_eos`, which is not
# what the harness sends. Reading the scenario settled it: it sets
# require_ignore_eos, and the load generator forwards nothing but --extra-inputs,
# which the harness sets to ignore_eos alone -- so no temperature field reaches
# the server and its OpenAI default of 1.0 applies. Production therefore samples,
# while the warm-up was taking the greedy argmax path and compiling neither the
# sampling kernels nor a decode batch that stayed populated.
#
# Two stages, and the second is the one that answers the question:
#
#   1. sweep the running container in both request shapes, filling the gap
#   2. throw that container away, boot a fresh one, and count what still compiles
#
# Stage 2 matters because stage 1 runs on a server that is already warm; only a
# cold container with a warm cache tests the claim that the mounts pay off across
# restarts.
#
# Usage: nohup finish_cache.sh &
set -u
ROOT=${ROOT:-/srv/contest-workspace}
BIN=${BIN:-$(cd "$(dirname "$0")" && pwd)}
LOG=${LOG:-$BIN/finish_cache.log}

caches() {
  for d in flashinfer_cache triton_cache inductor_cache humming_cache; do
    printf "    %-8s %6s  %5s files\n" "${d%_cache}" \
      "$(du -sh "$ROOT/$d" 2>/dev/null | cut -f1)" \
      "$(find "$ROOT/$d" -type f 2>/dev/null | wc -l)"
  done
}

jits() {
  echo "    reported during inference: $(docker logs llm-serve 2>&1 | grep -icE 'JIT compilation during inference')"
  docker logs llm-serve 2>&1 | grep -oE "JIT compilation during inference: [a-z_]+" \
    | sort | uniq -c | sed 's/^/      /'
}

{
  echo "=== stage 1: fill the missing request shapes on the running server  $(date -Is)"
  caches
  python3 "$BIN/warmup.py" --modes prod greedy || echo "WARMUP_FAILED stage1"
  echo "--- after stage 1:"
  caches
  jits

  echo
  echo "=== stage 2: fresh container, warm cache -- this is the verification  $(date -Is)"
  docker update --restart=no llm-serve >/dev/null 2>&1
  docker rm -f llm-serve >/dev/null 2>&1
  while docker ps -a --format '{{.Names}}' | grep -qx llm-serve; do sleep 2; done

  MNBT=16384 bash "$BIN/serve_kv.sh" fp8 || { echo "SERVE_FAILED stage2"; exit 1; }
  echo "--- boot log: is anything still being compiled at startup?"
  docker logs llm-serve 2>&1 | grep -iE "JIT-compiled|first run may take|Using configuration from" | head -5

  echo "--- a few real requests, production request shape:"
  python3 "$BIN/warmup.py" --modes prod || echo "WARMUP_FAILED stage2"
  jits
  echo "--- final caches:"
  caches
  echo "=== done $(date -Is)"
} > "$LOG" 2>&1
echo "FINISH_CACHE_DONE"
