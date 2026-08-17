#!/bin/bash
# Launch the inference server in one of two configurations and wait for /health.
#
#   base  reference configuration
#   opt   base plus a bfloat16 SSM state cache
#
# The SSM state cache dtype is the only difference between them. It matters more
# than its name suggests: with prefix caching on a hybrid attention/SSM model the
# attention page size is forced up to match the SSM page, so an fp32 state pushes
# the block to 4176 tokens while bfloat16 halves it to 2128. A smaller block
# wastes less of every partially-filled sequence, which is where the extra
# resident KV comes from.
#
#   --max-model-len 262144
#       The checkpoint's native maximum. Kept identical in both arms so the two
#       serve the same set of requests and the comparison stays about the cache.
#   --kv-cache-dtype fp8
#       Doubles resident KV tokens.
#   --kv-offloading-backend native --kv-offloading-size N
#       Host-RAM second tier. Long multi-turn sessions re-send most of their
#       prefix, so a RAM tier avoids re-reading it through the model. This build
#       backs the tier with a file under /dev/shm, so N must fit the tmpfs.
#   --max-num-batched-tokens 8192
#       Must be >= the attention block size or the scheduler asserts at boot.
#   --language-model-only --skip-mm-profiling
#       Text-only serving; skipping the vision tower avoids an OOM during
#       multimodal memory profiling.
#
# VLLM_TORCH_PROFILER_DIR only constructs the profiler; nothing is recorded until
# /start_profile is called, so it is safe to leave configured during a timed run.
#
# Do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here: it is incompatible
# with the offloading connector's pinned buffers and the engine dies at boot.
#
# Usage: serve_r2.sh <base|opt> [port]
set -u
ARM=${1:?usage: serve_r2.sh <base|opt> [port]}
PORT=${2:-8000}
ROOT=${ROOT:-/srv/contest-workspace}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/Qwen3.5-122B-A10B-FP8}
CACHE_DIR=${CACHE_DIR:-$ROOT/vllm_cache}
PROFILE_DIR=${PROFILE_DIR:-$ROOT/profiles}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.25.0}
NAME=${NAME:-llm-serve}
OFF=${OFF:-100}

ARGS="--max-model-len 262144 \
 --gpu-memory-utilization 0.96 \
 --language-model-only --skip-mm-profiling \
 --enable-prefix-caching --kv-cache-dtype fp8 \
 --max-num-batched-tokens 8192 --max-num-seqs 32 \
 --max-cudagraph-capture-size 128 --trust-remote-code \
 --kv-offloading-backend native --kv-offloading-size $OFF \
 --enable-prompt-tokens-details"

case "$ARM" in
  base) ;;
  opt)  ARGS="$ARGS --mamba-ssm-cache-dtype bfloat16" ;;
  *)    echo "unknown arm: $ARM"; exit 2 ;;
esac

mkdir -p "$CACHE_DIR" "$PROFILE_DIR"
docker rm -f "$NAME" >/dev/null 2>&1
# --ipc host makes --shm-size a no-op: the container sees the host's /dev/shm,
# which is what the offload tier is sized against.
docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --ulimit memlock=-1 \
  -v "$MODEL_DIR":/model:ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$PROFILE_DIR":/profiles \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_TORCH_PROFILER_DIR=/profiles \
  "$IMAGE" \
  --model /model --served-model-name Qwen3.5-122B-A10B-FP8 \
  --host 0.0.0.0 --port "$PORT" \
  $ARGS || exit 1

echo "launched $NAME arm=$ARM off=${OFF}GiB port=$PORT"
df -h /dev/shm

for i in $(seq 1 150); do
  if curl -s -m 3 http://127.0.0.1:"$PORT"/health >/dev/null 2>&1; then
    echo "READY after $((i*10))s"
    df -h /dev/shm; ls -la /dev/shm 2>/dev/null | head
    exit 0
  fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
    echo "CONTAINER DIED"; docker logs "$NAME" 2>&1 | tail -40; exit 1; }
  sleep 10
done
echo "TIMEOUT waiting for /health"; exit 2
