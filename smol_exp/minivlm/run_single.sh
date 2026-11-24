#!/bin/bash
# Per-image attack (sanity check, single_image.py): can ONE image be fooled with a small perturbation of its own?
#   ./run_single.sh qwen2-vl-2b              (also smolvlm-256m llava-ov-0.5b qwen2-vl-2b-224)
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./run_single.sh qwen2-vl-2b|qwen2-vl-2b-224|smolvlm-256m|llava-ov-0.5b [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
for M in "$@"; do
    mkdir -p results/$M
    echo "=== single_image.py --model $M ($(date)) ==="
    python single_image.py --model "$M" 2>&1 | tee -a results/$M/log_single.txt || { echo "!!! failed: $M"; exit 1; }
done
