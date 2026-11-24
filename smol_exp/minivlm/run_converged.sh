#!/bin/bash
# Every attack again, run to CONVERGENCE, for encoders whose mini-VLM is already trained (connectors are kept):
#   1. token v (as in the report) and connector-output v (what the language model receives), to convergence
#   2. COCO val and ImageNetV2 held-out captions (own v, connector v, random, transfers)
#   3. token damage with the transfers available so far
#   4. defense: no-defense / FT-fixed / FT-UAT, adaptive attacks to convergence (the encoders are fine-tuned again,
#      since they were trained against the old perturbation)
# Same split as before (fine-tuned encoders stay on the local disk of the machine):
#   espagne:   ./run_converged.sh ijepa clip
#   allemagne: ./run_converged.sh mae siglip
# Then relaunch the same commands once both are done (adds the missing transfers), then python compare.py.
# SMOLVLM=1 ./run_converged.sh ...  also runs smolvlm_reference.py at the end (on one machine only).
# The first run moves the old results (10-epoch perturbations) to results/old_10epochs/ and the old fine-tuned
# encoders of this machine to MINIVLM_DATA/defense/old_10epochs/. Nothing else may run on these results meanwhile.
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: [SMOLVLM=1] ./run_converged.sh siglip|clip|ijepa|mae [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false

python -c "import torch, torchvision, transformers, peft, sentence_transformers, pycocoevalcap" || {
    echo "!!! Missing packages: source ~/venv-universal/bin/activate && pip install -r requirements.txt"; exit 1; }

# Archive the old results of ALL encoders once (shared home), so that no old perturbation is used as a transfer
if [ ! -e results/.converged ]; then
    echo "=== Archiving the 10-epoch results to results/old_10epochs/ ==="
    for E in siglip clip ijepa mae; do
        mkdir -p results/old_10epochs/$E
        for f in uap_xi10.pt uap_xi10_ckpt.pt uap_xi10_history.json uap_xi10_tokens.json \
                 coco_stage2.json coco_captions_stage2.json eval_captions_stage2_xi10.json captions_stage2_xi10.json \
                 defense_ft.json defense_ft_captions.json log_uap.txt log_eval.txt log_defense.txt; do
            [ -e results/$E/$f ] && mv results/$E/$f results/old_10epochs/$E/
        done
        mv results/$E/adaptive_xi10_* results/old_10epochs/$E/ 2>/dev/null
    done
    touch results/.converged
fi
# Old fine-tuned encoders of this machine (trained against the old perturbation)
if [ ! -e "$MINIVLM_DATA/defense/.converged" ]; then
    mkdir -p "$MINIVLM_DATA/defense/old_10epochs"
    mv "$MINIVLM_DATA"/defense/*.pt "$MINIVLM_DATA/defense/old_10epochs/" 2>/dev/null
    touch "$MINIVLM_DATA/defense/.converged"
fi

for ENC in "$@"; do
    R=results/$ENC
    for f in connector_stage2.pt lora_stage2; do
        [ -e "$R/$f" ] || { echo "!!! $ENC: $R/$f missing, run ./run_minivlm.sh $ENC first"; exit 1; }
    done
    run() {  # run <log> <command...>
        log=$1; shift
        echo "=== $* ($(date)) ==="
        "$@" 2>&1 | tee -a "$R/$log" || { echo "!!! failed: $*"; exit 1; }
    }
    run log_uap.txt python uap_encoder.py --encoder "$ENC" --target tokens
    run log_uap.txt python uap_encoder.py --encoder "$ENC" --target connector
    run log_eval.txt python eval_coco.py --encoder "$ENC"
    run log_eval.txt python evaluate_captions.py --encoder "$ENC"
    run log_uap.txt python uap_encoder.py --encoder "$ENC" --target tokens      # token damage, transfers so far
    run log_defense.txt python defense_finetune.py --encoder "$ENC" --train_blocks "${TRAIN_BLOCKS:--1}"
    echo "=== $ENC done ($(date)) ==="
done
if [ "$SMOLVLM" = 1 ]; then
    echo "=== smolvlm_reference.py ($(date)) ==="
    mkdir -p results/smolvlm
    python smolvlm_reference.py 2>&1 | tee -a results/smolvlm/log.txt || echo "!!! smolvlm_reference.py failed"
fi
echo "=== Finished: $* ($(date)). When both machines are done: relaunch for the transfers, then python compare.py ==="
