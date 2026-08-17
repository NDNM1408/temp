#!/bin/bash
# One measurement arm end to end, detached: launch, capture a decode profile,
# replay the trace corpus, score, and dump the counters that explain the score.
#
# The profile is taken before the timed replay rather than during it: recording
# perturbs step time, and the question it answers -- which subsystem owns a
# decode step -- does not depend on the replay running.
#
# Usage: nohup run_arm.sh <base|opt> [concurrency] [duration] &
set -u
ARM=${1:?usage: run_arm.sh <base|opt> [concurrency] [duration]}
C=${2:-5}
D=${3:-900}
ROOT=${ROOT:-/srv/contest-workspace}
# The helpers travel together, so they are found next to this script wherever it
# was checked out rather than at a fixed path.
BIN=${BIN:-$(cd "$(dirname "$0")" && pwd)}
LOG=${LOG:-$BIN/arm_$ARM.log}
export PROFILE_DIR=${PROFILE_DIR:-$ROOT/profiles}

{
  echo "=== ARM $ARM C=$C D=$D started $(date -Is)"
  bash "$BIN/serve_r2.sh" "$ARM" || { echo "SERVE_FAILED"; exit 1; }

  echo "=== boot-time evidence"
  docker logs llm-serve 2>&1 | grep -iE \
    "cudaHostRegister|pin|offload|mmap|shm|KV cache size|GPU KV cache size|block size|Maximum concurrency" \
    | head -30
  df -h /dev/shm; ls -la /dev/shm | head

  echo "=== decode profile (15 streams over ~65k context)"
  python3 "$BIN/step_split.py" trace 15 65000 || echo "PROFILE_FAILED"

  echo "=== replay"
  bash "$BIN/bench_r2.sh" "$ARM" "$C" "$D"

  echo "=== score"
  python3 "$BIN/scorer.py" "$ROOT/bench/$ARM"

  echo "=== per-request TTFT anatomy"
  python3 "$BIN/ttft_anatomy.py" "$ROOT/bench/$ARM" || echo "ANATOMY_FAILED"
  python3 "$BIN/ttft_load.py" "$ROOT/bench/$ARM" || echo "LOAD_FAILED"

  echo "=== counters"
  grep -E 'prefix_cache_(queries|hits)_total|external_prefix_cache|kv_offload_total_bytes_total|num_preemptions_total' \
    "$ROOT/bench/$ARM/metrics_after.txt" | grep -v '^#'
  echo "=== ARM $ARM done $(date -Is)"
} > "$LOG" 2>&1
echo "ARM_DONE $ARM"
