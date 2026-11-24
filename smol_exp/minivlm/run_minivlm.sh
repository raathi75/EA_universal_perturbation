#!/bin/bash
# The whole experiment for one or several encoders, one after the other, on one machine.
# Encoders: siglip clip ijepa mae. On two machines (one ViT-H per machine to balance the time):
#   machine 1:  ./run_minivlm.sh ijepa clip
#   machine 2:  ./run_minivlm.sh mae siglip
# Results go to results/<encoder>/ (home directory, shared). Once everything is done, relaunch the same
# commands: they only add the transfers between encoders that were missing (a few minutes), then: python compare.py
# Can be relaunched at any time: finished steps are skipped, interrupted ones resume from their checkpoint.
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./run_minivlm.sh siglip|clip|ijepa|mae [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
# Model weights on the local disk too, not in the home quota
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false

python -c "import torch, torchvision, transformers, peft, sentence_transformers, pycocoevalcap" || {
    echo "!!! Missing packages: source ~/venv-universal/bin/activate && pip install -r requirements.txt"; exit 1; }

./prepare_data.sh || exit 1

for ENC in "$@"; do
    R=results/$ENC
    mkdir -p "$R"
    step() {  # step <file produced> <log> <command...>
        out=$1; log=$2; shift 2
        if [ -e "$out" ]; then echo "--- already done: $out"; return; fi
        echo "=== $* ($(date)) ==="
        "$@" 2>&1 | tee -a "$R/$log" || { echo "!!! failed: $*"; exit 1; }
    }
    always() {  # always <log> <command...>: incremental steps (skip what is done, add new transfers)
        log=$1; shift
        echo "=== $* ($(date)) ==="
        "$@" 2>&1 | tee -a "$R/$log" || { echo "!!! failed: $*"; exit 1; }
    }
    # 1. Universal perturbation on the encoder alone (ImageNetV2 build set; does not need the connector)
    step "$R/uap_xi10.pt" log_uap.txt python uap_encoder.py --encoder "$ENC"
    # 2. Connector, then connector + LoRA, on COCO train2017
    step "$R/connector_stage1.pt" log_train.txt python train_connector.py --encoder "$ENC" --stage 1
    step "$R/connector_stage2.pt" log_train.txt python train_connector.py --encoder "$ENC" --stage 2
    # 3. COCO val2017: caption quality and attack against the human captions
    always log_eval.txt python eval_coco.py --encoder "$ENC"
    # 4. ImageNetV2 held-out images of the SmolVLM report: captions changed
    always log_eval.txt python evaluate_captions.py --encoder "$ENC"
    # 5. Token damage, with the transfers available so far
    always log_uap.txt python uap_encoder.py --encoder "$ENC"
    echo "=== $ENC done ($(date)) ==="
done
echo "=== Finished: $* ($(date)). When all encoders are done: relaunch for the transfers, then python compare.py ==="
