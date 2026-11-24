#!/bin/bash
# Follow-up after run_all.sh: understand why the multi-layer loss wins.
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error
P=/c/Users/david/uap_env/Scripts/python.exe
L=results/pipeline.log
run() { echo "=== $(date +%H:%M:%S) START $*" >> $L; "$P" "$@" 2>&1 | grep -v -i "warn\|symlink\|Loading weights\|LOAD REPORT\|UNEXPECTED\|^Key \|^---\|^Notes\|position_ids\|^$" >> $L; echo "=== $(date +%H:%M:%S) END   $*" >> $L; }

run train.py multi_mid multi_all
run evaluate.py --model 256M --vs multi_mid multi_all --off 200 --layers 50
run robustness.py random group_final final h3 h6 h9 multi multi_mid multi_all
run evaluate.py --model 500M --vs multi_mid multi_all --off 200 --layers 50
run transfer_other.py classifiers --vs multi_mid multi_all
run evaluate.py --model 256M --vs multi_mid multi_all --on 100 --off 0 --layers 0
echo "=== $(date +%H:%M:%S) EXTRA DONE" >> $L
