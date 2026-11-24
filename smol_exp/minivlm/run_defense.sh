#!/bin/bash
# Defense by fine-tuning the vision encoder (recipe of the SmolVLM report, notebook 07), for one or several
# encoders whose mini-VLM is already trained (run_minivlm.sh done). Same split as run_minivlm.sh:
#   machine 1:  ./run_defense.sh ijepa clip
#   machine 2:  ./run_defense.sh mae siglip
# Fine-tuned encoders are kept on the local disk of the machine (MINIVLM_DATA/defense): run each encoder on
# one machine only. Can be relaunched: finished steps are skipped.
# Out of GPU memory: TRAIN_BLOCKS=8 ./run_defense.sh ...  (only the last 8 blocks are fine-tuned)
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./run_defense.sh siglip|clip|ijepa|mae [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
for ENC in "$@"; do
    R=results/$ENC
    for f in uap_xi10.pt connector_stage2.pt; do
        [ -e "$R/$f" ] || { echo "!!! $ENC: $R/$f missing, run ./run_minivlm.sh $ENC first"; exit 1; }
    done
    echo "=== defense_finetune.py --encoder $ENC ($(date)) ==="
    python defense_finetune.py --encoder "$ENC" --train_blocks "${TRAIN_BLOCKS:--1}" 2>&1 | tee -a "$R/log_defense.txt" \
        || { echo "!!! failed: $ENC"; exit 1; }
done
echo "=== Defense finished: $* ($(date)) ==="
