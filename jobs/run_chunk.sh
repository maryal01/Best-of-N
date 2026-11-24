#!/bin/bash
# Runs ONE chunk of initialize_experiment.py on the GPU(s) this shell can see.
# No Slurm inside, so the same script is used by jobs/gilbreth.sub (Slurm) and by
# jobs/run_local.sh (plain GPU workstations).
#
#   bash jobs/run_chunk.sh <dataset> <chunk index>        dataset: gsm8k | math500 | math
#
# Environment (all optional):
#   PY     python with vllm etc.             default: $BASE/bon-env/bin/python
#   BASE   big non-home dir for caches/answers   default: $RCAC_SCRATCH
#   CHUNK  questions per chunk                default: 100 (math, math500), 200 (gsm8k)
# Output: results/<prefix><k>.csv, generated answers in $BASE/bon_answers/ (reused if the
# chunk is rerun, e.g. after a preemption, so only the scoring is repeated).
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root

export DATASET=${1:?dataset: gsm8k | math500 | math}
TASK=${2:?chunk index}
BASE=${BASE:-${RCAC_SCRATCH:?set BASE or RCAC_SCRATCH}}

case $DATASET in
    gsm8k)
        N_QUESTIONS=1319; CHUNK=${CHUNK:-200}
        OUT_PREFIX=results/table_256_part ;;
    math500)
        N_QUESTIONS=500; CHUNK=${CHUNK:-100}
        OUT_PREFIX=results/math500_part
        export MAX_TOKENS=2048
        export SCORERS=Skywork/Skywork-Reward-V2-Qwen3-0.6B,OpenAssistant/reward-model-deberta-v3-large-v2 ;;
    math)
        N_QUESTIONS=4500; CHUNK=${CHUNK:-100}
        OUT_PREFIX=results/math_part
        export MAX_TOKENS=2048
        export SCORERS=Skywork/Skywork-Reward-V2-Qwen3-0.6B ;;
    *)
        echo "Unknown dataset '$DATASET'; use gsm8k, math500 or math." >&2
        exit 1 ;;
esac
export START=$((TASK * CHUNK)) END=$(((TASK + 1) * CHUNK))
if (( START >= N_QUESTIONS )); then
    echo "Chunk $TASK starts at question $START, past the $N_QUESTIONS in $DATASET; nothing to do."
    exit 0
fi

PY=${PY:-$BASE/bon-env/bin/python}
[[ -x $PY ]] || { echo "No Python at $PY; run jobs/setup_env.sh first." >&2; exit 1; }

# Everything big goes under BASE, never $HOME (home quotas are small).
export HF_HOME=$BASE/hf_cache
export VLLM_CACHE_ROOT=$BASE/vllm_cache
export PIP_USER=0 PYTHONNOUSERSITE=1
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$BASE/bon_answers" results

# Kernel compile caches default to ~/.triton etc. on NFS, shared by every task; tasks starting
# together then crash with "Stale file handle". Each chunk gets its own copy on local disk.
LOCAL_CACHE=${TMPDIR:-/tmp}/bon-$DATASET-$TASK-$$
export TRITON_CACHE_DIR=$LOCAL_CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$LOCAL_CACHE/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
trap 'rm -rf "$LOCAL_CACHE"' EXIT

echo "$DATASET chunk $TASK on $(hostname), started $(date), python: $PY"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Fail in seconds, not after holding a GPU, if a package is missing or torch has no GPU.
"$PY" - <<'EOF'
import os, sys
from vllm import LLM, SamplingParams
import transformers, accelerate, datasets, torch
if os.environ["DATASET"] in ("math500", "math"):
    import math_verify
if not torch.cuda.is_available():
    sys.exit("torch cannot use the GPU")
print("packages OK: torch", torch.__version__, "on", torch.cuda.get_device_name(0))
EOF

export OUT=$OUT_PREFIX$TASK.csv
export ANSWERS=$BASE/bon_answers/${DATASET}_part$TASK.jsonl
echo "questions $START..$((END - 1)) -> $OUT (answers: $ANSWERS)"
"$PY" -u initialize_experiment.py
echo "Finished $(date)"
