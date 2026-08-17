#!/bin/bash
# Replay the agent-session trace corpus against a local server and export one
# record per request.
#
#   --max-context-length 204800
#       Client-side filter. A trace whose peak (prompt + that turn's max_tokens)
#       exceeds this is dropped before replay, so this decides which sessions are
#       in the pool at all, not just how many. It is meant to mirror what the
#       server can actually serve.
#   --concurrency 5
#       Number of session trees kept live for the whole run.
#   --use-server-token-count
#       Token counts come from the server's own usage block rather than a local
#       re-tokenisation, which is what makes the cache-read counts trustworthy.
#   --export-level records
#       Needed for any per-request analysis; the summary alone cannot separate
#       queueing from prefill.
#
# Counters are snapshotted either side of the replay so cache and offload traffic
# can be attributed to this run rather than to warm-up.
#
# Usage: bench_r2.sh <name> [concurrency] [duration]
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/srv/contest-workspace/hf
N=${1:-run}; C=${2:-5}; D=${3:-900}
PORT=${PORT:-8000}
DS=${DS:-semianalysis_cc_traces_weka_062126}
MAXCTX=${MAXCTX:-204800}
ROOT=${ROOT:-/srv/contest-workspace}
OUT=$ROOT/bench/$N
mkdir -p "$OUT"

curl -s -m 10 http://127.0.0.1:$PORT/metrics > "$OUT/metrics_before.txt"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://127.0.0.1:$PORT \
  --endpoint-type chat \
  --model Qwen3.5-122B-A10B-FP8 \
  --tokenizer $ROOT/models/Qwen3.5-122B-A10B-FP8 \
  --public-dataset $DS \
  --max-context-length $MAXCTX \
  --concurrency $C \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --benchmark-duration $D \
  --random-seed 20260707 \
  --artifact-dir "$OUT" \
  --export-level records \
  --ui simple
RC=$?

curl -s -m 10 http://127.0.0.1:$PORT/metrics > "$OUT/metrics_after.txt"
echo "BENCH_DONE rc=$RC name=$N C=$C D=$D maxctx=$MAXCTX ds=$DS"
