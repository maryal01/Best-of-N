#!/bin/bash
# Evaluate generated code solutions with EvalPlus (base + plus tests) inside a bubblewrap sandbox.
# The model-written programs are untrusted, so the whole pipeline runs with:
#   - the file system read-only, and /home, /data, /scratch, /root, /mnt, /srv emptied, so nothing of ours or of other
#     users is visible; only the python env, the EvalPlus cache and the samples folder are bound back in
#   - no network, its own process and IPC namespaces, and everything killed if this script dies
#   - EvalPlus' own per-solution time and memory limits
#
#   BASE=<dir holding eval-env/, uv_python/ and cache/ (EvalPlus problems cache)> \
#   bash experiments/coding/evaluate_code.sh <mbpp|humaneval> <samples.jsonl> [parallel workers]
#
# samples.jsonl lines: {"task_id", "sample_id", "solution"}. Writes next to it: <name>-sanitized.jsonl
# (code extracted from the replies) and <name>-sanitized_eval_results.json (per solution base/plus status).
set -euo pipefail
DATASET=${1:?mbpp or humaneval}
SAMPLES=$(realpath "${2:?samples.jsonl}")
PARALLEL=${3:-16}
BASE=${BASE:?set BASE}
WORK=$(dirname "$SAMPLES")
NAME=$(basename "$SAMPLES" .jsonl)

hide=()                                          # empty out the big shared directories that exist on this machine
for d in /home /root /data /scratch /mnt /srv /media; do [[ -d $d ]] && hide+=(--tmpfs "$d"); done

sandbox() {
    bwrap --die-with-parent --unshare-net --unshare-pid --unshare-ipc \
        --ro-bind / / --dev /dev --proc /proc "${hide[@]}" --tmpfs /tmp \
        --ro-bind "$BASE/eval-env" "$BASE/eval-env" --ro-bind "$BASE/uv_python" "$BASE/uv_python" \
        --bind "$BASE/cache" "$BASE/cache" --bind "$WORK" "$WORK" \
        --chdir "$WORK" --setenv XDG_CACHE_HOME "$BASE/cache" --setenv HOME /tmp --setenv PYTHONNOUSERSITE 1 \
        "$@"
}

PY=$BASE/eval-env/bin/python
sandbox "$PY" -m evalplus.sanitize "$SAMPLES" --mbpp_version default
sandbox "$PY" -m evalplus.evaluate "$DATASET" --samples "$WORK/$NAME-sanitized.jsonl" --parallel "$PARALLEL" --i_just_wanna_run
