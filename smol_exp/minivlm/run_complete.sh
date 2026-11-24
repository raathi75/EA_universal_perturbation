#!/bin/bash
# Completes the data listed in RAPPORT.md (sections 4.7 and 6). One block per machine, run in tmux:
#   allemagne:  ./run_complete.sh allemagne      espagne:  ./run_complete.sh espagne
# Then, once BOTH are done, on either machine:   ./run_complete.sh final
# Every step can be relaunched (finished steps are skipped, attacks resume from their checkpoint).
# Order inside a block = priority: if time runs out, the most important data is already there.
#
#   1. environment of the machine (pip freeze, GPU, git commit) -> results/env_<host>.txt        [audit 4.1]
#   2. mechanism of the defense failure (connector-output cosine), needs the fine-tuned encoders [IV.3]
#   3. variance of the attack: own token v with seeds 1 and 2, + 4 extra random draws             [4.5]
#   4. missing transfers to MAE, end-to-end attacks of I-JEPA / MAE / SmolVLM / LLaVA            [4.7]
#   5. Qwen2-VL 448 end-to-end attack with the small step that worked at 224                    [VI.6]
set -o pipefail
cd "$(dirname "$0")"
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
SEEDS=${SEEDS:-"1 2"}

run() {  # run <log> <command...>
    log=$1; shift
    echo "=== $* ($(date)) ===" | tee -a "$log"
    "$@" 2>&1 | tee -a "$log" || { echo "!!! failed: $*"; exit 1; }
}
env_log() {
    f=results/env_$(hostname).txt
    { date; hostname; nvidia-smi; python --version; pip freeze; git rev-parse HEAD 2>/dev/null || echo "no git commit"; } > "$f" 2>&1
    echo ">> environment saved to $f"
}
mini() {  # mini <encoder>: seeds, then the evaluations (they add the new seeds, random draws and transfers)
    E=$1
    for S in $SEEDS; do run results/$E/log_uap.txt python uap_encoder.py --encoder "$E" --target tokens --seed "$S"; done
    run results/$E/log_eval.txt python eval_coco.py --encoder "$E"
    run results/$E/log_eval.txt python evaluate_captions.py --encoder "$E"
    run results/$E/log_uap.txt python uap_encoder.py --encoder "$E" --target tokens     # transfers to this encoder
}
real() {  # real <model>: seeds of the token v, then evaluation of everything new
    M=$1
    for S in $SEEDS; do run results/$M/log.txt python real_vlm.py --model "$M" --seed "$S"; done
}

case $1 in
    allemagne)   # fine-tuned siglip and mae are on this machine; Qwen2-VL
        env_log
        run results/siglip/log_defense.txt python defense_mechanism.py --encoder siglip
        run results/mae/log_defense.txt python defense_mechanism.py --encoder mae
        mini siglip
        mini mae                                   # also adds the transfers siglip -> mae and clip -> mae
        run results/mae/log_uap.txt python uap_encoder.py --encoder mae --target caption --lr 4
        run results/mae/log_eval.txt python eval_coco.py --encoder mae
        run results/mae/log_eval.txt python evaluate_captions.py --encoder mae
        real qwen2-vl-2b-224
        if [ ! -e results/qwen2-vl-2b/.lr05 ]; then    # redo the 448 end-to-end attack once, with the small step
            mkdir -p results/qwen2-vl-2b/old_lr4 && mv results/qwen2-vl-2b/uap_cap_xi10* results/qwen2-vl-2b/old_lr4/ 2>/dev/null
            python - <<'PY'   # drop its old caption results so that they are recomputed with the new v
import json
from pathlib import Path
d = Path('results/qwen2-vl-2b')
for f in ['coco_stage2.json', 'eval_captions_stage2_xi10.json', 'coco_captions_stage2.json', 'captions_stage2_xi10.json']:
    p = d / f
    j = json.load(open(p))
    for k in [k for k in list(j.get('results', j)) if 'caption loss' in k]:
        (j['results'] if 'results' in j else j).pop(k)
    json.dump(j, open(p, 'w'), indent=1)
PY
            touch results/qwen2-vl-2b/.lr05
        fi
        run results/qwen2-vl-2b/log.txt python real_vlm.py --model qwen2-vl-2b --target caption --lr 0.5
        real qwen2-vl-2b ;;
    espagne)     # fine-tuned clip and ijepa are on this machine; SmolVLM, LLaVA
        env_log
        run results/clip/log_defense.txt python defense_mechanism.py --encoder clip
        run results/ijepa/log_defense.txt python defense_mechanism.py --encoder ijepa
        mini clip
        mini ijepa
        run results/ijepa/log_uap.txt python uap_encoder.py --encoder ijepa --target caption --lr 4
        run results/ijepa/log_eval.txt python eval_coco.py --encoder ijepa
        run results/ijepa/log_eval.txt python evaluate_captions.py --encoder ijepa
        for M in smolvlm-256m llava-ov-0.5b; do
            # LLaVA ran out of GPU memory with batches of 8 for the end-to-end attack: 4 (same images, same epochs)
            run results/$M/log.txt python real_vlm.py --model "$M" --target caption --lr 1 --attack_batch_size 4
            real "$M"
        done ;;
    llava)       # the LLaVA part of espagne, on any machine with the data (e.g. allemagne once its block is done)
        run results/llava-ov-0.5b/log.txt python real_vlm.py --model llava-ov-0.5b --target caption --lr 1 --attack_batch_size 4
        real llava-ov-0.5b ;;
    final)       # tables (after both machines)
        python imagenet_accuracy.py | tail -1
        python compare.py > /dev/null && echo ">> results/comparison.md updated" ;;
    *) echo "usage: ./run_complete.sh allemagne|espagne|llava|final   (SEEDS='1 2' by default)"; exit 1 ;;
esac
echo "=== Finished: $1 ($(date)) ==="
