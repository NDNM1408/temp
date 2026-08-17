#!/bin/bash
# Run the attention probe inside the serving image.
#
# The probe needs torch, flashinfer and vllm at the exact versions the server
# runs, so it goes in the same image rather than a separate environment -- a
# kernel measured against a different flashinfer build answers a question nobody
# asked.
#
# It wants the GPU to itself. A live server holds ~137 GB and, more importantly,
# competes for SMs, which is exactly the quantity being measured. The check below
# refuses to start rather than quietly producing numbers that are 30% low.
#
# Usage:
#   bash run_attn_probe.sh                       # all arms
#   bash run_attn_probe.sh --backends flashinfer --kv-lens 8192 73728
#   FORCE=1 bash run_attn_probe.sh               # run anyway, numbers are noisy
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE=${PROBE:-$HERE/attn_probe.py}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.25.0}
OUT=${OUT:-$HERE/attn_probe.json}
FORCE=${FORCE:-0}

[ -f "$PROBE" ] || { echo "not found: $PROBE"; exit 1; }

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt 2000 ] && [ "$FORCE" != "1" ]; then
  echo "GPU has ${USED} MiB in use — something is still resident."
  docker ps --format '  {{.Names}} {{.Status}}'
  echo "Stop it first, or re-run with FORCE=1 and treat the numbers as a lower bound."
  exit 2
fi

docker run --rm --gpus all --ipc host \
  -v "$PROBE":/probe.py:ro \
  -v "$(dirname "$OUT")":/out \
  --entrypoint python3 \
  "$IMAGE" /probe.py --out "/out/$(basename "$OUT")" "$@"
RC=$?
echo "PROBE_DONE rc=$RC out=$OUT"
exit $RC
