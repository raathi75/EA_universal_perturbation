#!/bin/bash
# Full re-run pipeline, most important results first (laptop may be on battery).
cd "$(dirname "$0")"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error
P=/c/Users/david/uap_env/Scripts/python.exe
L=results/pipeline.log
run() { echo "=== $(date +%H:%M:%S) START $*" >> $L; "$P" "$@" 2>&1 | grep -v -i "warn\|symlink\|Loading weights\|LOAD REPORT\|UNEXPECTED\|^Key \|^---\|^Notes\|position_ids\|^$" >> $L; echo "=== $(date +%H:%M:%S) END   $*" >> $L; }

MAIN="final h3 h6 h9 multi"
run train.py $MAIN
run evaluate.py --model 256M --vs $MAIN --off 200 --layers 50
run transfer_other.py similarity
run evaluate.py --model 500M --vs group_final random $MAIN --off 200 --layers 50
run transfer_other.py classifiers --vs group_final $MAIN
EXTRA="h2 h4 h8 h10 h12"
run train.py $EXTRA
run evaluate.py --model 256M --vs $EXTRA --off 200 --layers 50
run evaluate.py --model 256M --vs group_final random $MAIN --on 100 --off 0 --layers 0
run evaluate.py --model 500M --vs $EXTRA --off 200 --layers 50
run transfer_other.py classifiers --vs $EXTRA
run evaluate.py --model 2.2B --vs group_final random $MAIN --off 100 --layers 0 --bs 2
echo "=== $(date +%H:%M:%S) ALL DONE" >> $L
