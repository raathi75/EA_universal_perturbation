#!/bin/bash
# Post-training with the encoder UNFROZEN (as in SmolVLM) vs frozen (the main mini-VLMs): same stage 1 (copied),
# stage 2 = connector + LoRA + encoder (lr 2e-6). Then the attacks, to convergence:
#   fixed:    the perturbation of the ORIGINAL encoder (built on the frozen mini-VLM, run_converged.sh), unchanged
#   adaptive: a perturbation recomputed against the post-trained encoder (tokens, and connector outputs)
# Run after run_converged.sh for the same encoders (the fixed attack is its perturbation). The post-trained encoder
# stays on the local disk: run each encoder on one machine only.
#   espagne:   ./run_unfrozen.sh ijepa clip        allemagne: ./run_unfrozen.sh mae siglip
# Out of GPU memory: TRAIN_BLOCKS=8 ./run_unfrozen.sh ...   (only the last 8 blocks of the encoder are trained)
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./run_unfrozen.sh siglip|clip|ijepa|mae [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
for ENC in "$@"; do
    M=$ENC-unfrozen
    R=results/$M
    mkdir -p "$R"
    [ -e results/$ENC/connector_stage1.pt ] || { echo "!!! results/$ENC/connector_stage1.pt missing: ./run_minivlm.sh $ENC first"; exit 1; }
    [ -e results/$ENC/uap_xi10.pt ] || echo "!!! results/$ENC/uap_xi10.pt missing: the fixed attack will be skipped (run_converged.sh first)"
    step() {  # step <file produced> <log> <command...>
        out=$1; log=$2; shift 2
        if [ -e "$out" ]; then echo "--- already done: $out"; return; fi
        echo "=== $* ($(date)) ==="
        "$@" 2>&1 | tee -a "$R/$log" || { echo "!!! failed: $*"; exit 1; }
    }
    run() {
        log=$1; shift
        echo "=== $* ($(date)) ==="
        "$@" 2>&1 | tee -a "$R/$log" || { echo "!!! failed: $*"; exit 1; }
    }
    step "$R/connector_stage1.pt" log_train.txt python train_connector.py --encoder "$M" --stage 1
    step "$R/connector_stage2.pt" log_train.txt python train_connector.py --encoder "$M" --stage 2 \
        --save_every 1000 --train_blocks "${TRAIN_BLOCKS:--1}"
    run log_uap.txt python uap_encoder.py --encoder "$M" --target tokens
    run log_uap.txt python uap_encoder.py --encoder "$M" --target connector
    run log_eval.txt python eval_coco.py --encoder "$M"
    run log_eval.txt python evaluate_captions.py --encoder "$M"
    echo "=== $M done ($(date)) ==="
done
echo "=== Finished: $* ($(date)). Then: python compare.py ==="
