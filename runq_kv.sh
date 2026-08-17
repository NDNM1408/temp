#!/bin/bash
# Two arms, one variable: the KV cache dtype.
#
#   B  token budget 8192, KV fp8    -- the configuration that was last scored,
#                                      so it doubles as a repeat measurement
#                                      against a known reference
#   C  token budget 8192, KV bf16   -- identical except the dtype
#
# Both arms sit at 8192 rather than the 16384 currently in production, because
# 16384 costs about 15% of the KV pool and mixing that in would confound the
# comparison. Arm B therefore also answers, by comparison with production,
# whether the larger budget was worth its pool.
#
# Counters are snapshotted immediately after each replay and before anything is
# restarted. The last time this was skipped, the container was recycled and the
# only record of a scored run was gone.
#
# Usage: nohup runq_kv.sh [duration] &
set -u
D=${1:-900}
C_LEVEL=${C_LEVEL:-5}
ROOT=${ROOT:-/srv/contest-workspace}
BIN=${BIN:-$(cd "$(dirname "$0")" && pwd)}
OUTDIR=${OUTDIR:-$BIN/runq_kv}
mkdir -p "$OUTDIR"

run_arm() {
  local arm=$1 dtype=$2
  local log=$OUTDIR/$arm.log
  {
    echo "=== ARM $arm dtype=$dtype mnbt=8192 C=$C_LEVEL D=${D}s  $(date -Is)"
    MNBT=8192 bash "$BIN/serve_kv.sh" "$dtype" || { echo "SERVE_FAILED"; return 1; }

    echo "=== warm-up (pays the JIT and autotune, and prices it)"
    python3 "$BIN/warmup.py" || echo "WARMUP_FAILED"

    echo "=== baseline counters (warm-up traffic is in these; subtract before reading hit rates)"
    curl -s -m 15 http://127.0.0.1:8000/metrics > "$OUTDIR/$arm.metrics_baseline.txt"

    echo "=== replay"
    bash "$BIN/bench_r2.sh" "kv_$arm" "$C_LEVEL" "$D"

    echo "=== counters, before anything restarts"
    curl -s -m 15 http://127.0.0.1:8000/metrics > "$OUTDIR/$arm.metrics_after.txt"

    echo "=== score"
    python3 "$BIN/scorer.py" "$ROOT/bench/kv_$arm"

    echo "=== what this arm traded"
    docker logs llm-serve 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | head -2
    grep -E '^vllm:num_preemptions_total|^vllm:prefix_cache_(queries|hits)_total' \
      "$OUTDIR/$arm.metrics_after.txt" | grep -v '^#'

    echo "=== ARM $arm done $(date -Is)"
  } > "$log" 2>&1
}

run_arm B fp8
run_arm C bf16

echo "== summary =="
for a in B C; do
  echo "--- arm $a"
  grep -E "ERS =|mean s_ttft|mean s_tpot|frac TTFT>6000|TTFT ms mean|TPOT ms mean|GPU KV cache size|num_preemptions_total|one-time compile" \
    "$OUTDIR/$a.log" | head -12
done
echo "RUNQ_KV_DONE"
