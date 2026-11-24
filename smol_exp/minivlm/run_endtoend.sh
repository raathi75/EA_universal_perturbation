#!/bin/bash
# END-TO-END universal perturbation: instead of the token cosine (the label-free proxy of the report), the attack
# lowers the likelihood of each model's own clean caption of the build images, through the whole model (encoder,
# projector, language model). If a model resists this attack too, it is really robust; if not, its robustness
# under the token attack was a weakness of the proxy. Bigger first step (LR, on the 0-255 scale, ξ = 10) to go
# faster: the step is halved on every plateau and the best v on the build images is kept, so a too large step
# only costs an epoch or two.
#   real VLMs:  ./run_endtoend.sh qwen2-vl-2b            (also smolvlm-256m llava-ov-0.5b)
#   mini-VLMs:  ./run_endtoend.sh siglip                  (also clip ijepa mae siglip-unfrozen)
# Then python compare.py (rows "end-to-end v"). Relaunch = resume from the checkpoint.
set -o pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: [LR=4] ./run_endtoend.sh qwen2-vl-2b|qwen2-vl-2b-224|smolvlm-256m|llava-ov-0.5b|siglip|clip|ijepa|mae [...]"; exit 1; }
export MINIVLM_DATA=${MINIVLM_DATA:-/Data/gurvan.nkouathio-kemnhou/minivlm}
export HF_HOME=${HF_HOME:-$MINIVLM_DATA/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
LR=${LR:-4}
run() {  # run <log> <command...>
    log=$1; shift
    echo "=== $* ($(date)) ==="
    "$@" 2>&1 | tee -a "$log" || { echo "!!! failed: $*"; exit 1; }
}
for M in "$@"; do
    mkdir -p results/$M
    case $M in
        smolvlm-256m|llava-ov-0.5b|qwen2-vl-2b|qwen2-vl-2b-224)
            # attack, then the captions of the new condition (the other conditions are already done)
            run results/$M/log.txt python real_vlm.py --model "$M" --target caption --lr "$LR" ;;
        *)
            run results/$M/log_uap.txt python uap_encoder.py --encoder "$M" --target caption --lr "$LR"
            run results/$M/log_eval.txt python eval_coco.py --encoder "$M"
            run results/$M/log_eval.txt python evaluate_captions.py --encoder "$M" ;;
    esac
done
python imagenet_accuracy.py | tail -1
python compare.py > /dev/null && echo ">> results/comparison.md updated"
echo "=== Finished: $* ($(date)) ==="
