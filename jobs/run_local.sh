#!/bin/bash
# Run several chunks one after another on ONE GPU of a plain (non-Slurm) machine, e.g. porsche
# or ferrari. Start one copy per GPU you want to use, each with its own list of chunks.
#
#   BASE=/path/with/lots/of/space PY=$BASE/bon-env/bin/python \
#   nohup bash jobs/run_local.sh <dataset> <gpu id> <chunk> [<chunk> ...] > logs/gpu0.log 2>&1 &
#
# Chunks whose result file already exists are skipped, so the script can simply be restarted.
set -uo pipefail
cd "$(dirname "$0")/.."
DATASET=${1:?dataset}; GPU=${2:?gpu id}; shift 2
mkdir -p logs

prefix() { case $1 in gsm8k) echo results/table_256_part;; math500) echo results/math500_part;; math) echo results/math_part;; esac; }
# math500 writes one file per scorer, so check for the Skywork one; the others have a single file.
done_file() { [[ $1 == math500 ]] && echo "$(prefix $1)$2_Skywork-Reward-V2-Qwen3-0.6B.csv" || echo "$(prefix $1)$2.csv"; }

for task in "$@"; do
    if [[ -s $(done_file "$DATASET" "$task") ]]; then echo "chunk $task already done, skipping"; continue; fi
    CUDA_VISIBLE_DEVICES=$GPU bash jobs/run_chunk.sh "$DATASET" "$task" > "logs/${DATASET}_chunk${task}_gpu${GPU}.log" 2>&1 \
        && echo "chunk $task done $(date)" || echo "chunk $task FAILED $(date) (see logs/${DATASET}_chunk${task}_gpu${GPU}.log)"
done
