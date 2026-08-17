#!/bin/bash
# Launch the server with the KV cache dtype as the only variable.
#
# Everything else is held at the configuration currently in production, so a
# difference between two runs of this script can only be attributed to the dtype.
# In particular the token budget stays at 8192: raising it to 16384 costs about
# 15% of the KV pool (1,300,637 -> 1,105,037 tokens) because the activation
# buffers grow, and mixing that into a dtype comparison would confound the two
# things the comparison is meant to separate.
#
#   fp8    what production runs. Halves KV bytes, so the pool holds twice as many
#          tokens. On this card the fp8 attention path is also suspected of being
#          slower than bf16 at head_dim 256.
#   bf16   the model's own dtype. Attention measured 5.8x faster in an isolated
#          microbenchmark, but the pool halves to roughly 650k tokens, and five
#          live session trees need 471k at the median and 737k at the upper
#          quartile -- so this arm is expected to evict, and whether the faster
#          kernel pays for the eviction is the entire question.
#
# Usage: serve_kv.sh <fp8|bf16> [port]
set -u
DTYPE=${1:?usage: serve_kv.sh <fp8|bf16> [port]}
PORT=${2:-8000}
ROOT=${ROOT:-/srv/contest-workspace}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/Qwen3.5-122B-A10B-FP8}
CACHE_DIR=${CACHE_DIR:-$ROOT/vllm_cache}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.25.0}
NAME=${NAME:-llm-serve}
MNBT=${MNBT:-8192}

case "$DTYPE" in
  fp8)  KV_FLAG="--kv-cache-dtype fp8" ;;
  bf16) KV_FLAG="--kv-cache-dtype auto" ;;   # auto resolves to the model dtype
  *)    echo "unknown dtype: $DTYPE (want fp8 or bf16)"; exit 2 ;;
esac

ARGS="--max-model-len 262144 \
 --gpu-memory-utilization 0.96 \
 --language-model-only --skip-mm-profiling \
 --enable-prefix-caching --prefix-caching-hash-algo xxhash \
 $KV_FLAG \
 --mamba-ssm-cache-dtype bfloat16 \
 --attention-backend FLASHINFER \
 --max-num-batched-tokens $MNBT --max-num-seqs 32 \
 --max-cudagraph-capture-size 128 --trust-remote-code \
 --enable-prompt-tokens-details"

mkdir -p "$CACHE_DIR"
# A restart policy on an earlier container races with removal: the daemon brings
# it back between the kill and the remove, and the next `docker run` then fails
# on a name conflict.
docker update --restart=no "$NAME" >/dev/null 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
while docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; do sleep 2; done

docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --ulimit memlock=-1 \
  -v "$MODEL_DIR":/model:ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -e VLLM_LOGGING_LEVEL=INFO \
  "$IMAGE" \
  --model /model --served-model-name Qwen3.5-122B-A10B-FP8 \
  --host 0.0.0.0 --port "$PORT" \
  $ARGS || exit 1

echo "launched $NAME dtype=$DTYPE mnbt=$MNBT port=$PORT"
for i in $(seq 1 150); do
  if curl -s -m 3 http://127.0.0.1:"$PORT"/health >/dev/null 2>&1; then
    echo "READY after $((i*10))s"
    # The pool size is the number this arm trades against, so record it next to
    # the score rather than leaving it in a log nobody re-reads.
    docker logs "$NAME" 2>&1 | grep -E "GPU KV cache size|Maximum concurrency|kv_cache_dtype" | head -3
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
    echo "CONTAINER DIED"; docker logs "$NAME" 2>&1 | tail -40; exit 1; }
  sleep 10
done
echo "TIMEOUT waiting for /health"; exit 2
