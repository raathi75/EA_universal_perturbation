#!/bin/bash
# Real pre-trained VLMs with the protocol of the mini-VLMs (real_vlm.py): own universal v to convergence, captions on
# COCO val and ImageNetV2 held-out, transfers. Models: smolvlm-256m llava-ov-0.5b qwen2-vl-2b qwen2-vl-2b-224. Any machine; relaunch
# once all are done to add the transfers between them, then python compare.py.
#   e.g. espagne: ./run_real_vlm.sh llava-ov-0.5b smolvlm-256m      allemagne: ./run_real_vlm.sh qwen2-vl-2b
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./run_real_vlm.sh smolvlm-256m|llava-ov-0.5b|qwen2-vl-2b|qwen2-vl-2b-224 [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
for M in "$@"; do
    mkdir -p results/$M
    echo "=== real_vlm.py --model $M ($(date)) ==="
    python real_vlm.py --model "$M" 2>&1 | tee -a results/$M/log.txt || { echo "!!! failed: $M"; exit 1; }
done
echo "=== Finished: $* ($(date)) ==="
