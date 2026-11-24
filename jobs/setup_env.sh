#!/bin/bash
# One-time environment setup: Python 3.12 venv with vLLM etc. Everything (venv, uv, pip/uv
# caches, Python build, temp files, HF cache) goes under BASE, NOTHING is written to $HOME.
#   BASE=$RCAC_SCRATCH bash jobs/setup_env.sh
# Result: $BASE/bon-env/bin/python   (use it as PY in the job scripts)
# Optional: VLLM_SPEC="vllm==0.11.0"  UV_ARGS="--torch-backend=cu128"  for machines whose NVIDIA driver is
# too old for the newest torch build (drivers older than about 580 need a cu12x build; a recent driver works with the default).
set -euo pipefail
BASE=${BASE:?set BASE to a big non-home directory, e.g. \$RCAC_SCRATCH or a /data/\$USER folder}
mkdir -p "$BASE"/{uv-bootstrap,uv_cache,uv_python,pip_cache,tmp,hf_cache}
export UV_CACHE_DIR=$BASE/uv_cache UV_PYTHON_INSTALL_DIR=$BASE/uv_python
export PIP_CACHE_DIR=$BASE/pip_cache TMPDIR=$BASE/tmp HF_HOME=$BASE/hf_cache
export PIP_USER=0 PYTHONNOUSERSITE=1

# uv itself. Preferably pip-installed into a small bootstrap venv; if the system python has no
# venv/pip (some plain workstations), download uv's standalone release binary instead.
UV=$BASE/uv-bootstrap/bin/uv
if [[ ! -x $UV ]]; then
    if python3 -m venv "$BASE/uv-bootstrap" 2>/dev/null && "$BASE/uv-bootstrap/bin/pip" install -q uv; then :
    else
        rm -rf "$BASE/uv-bootstrap"; mkdir -p "$BASE/uv-bootstrap/bin"
        curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
            | tar -xz --strip-components=1 -C "$BASE/uv-bootstrap/bin"
    fi
fi

[[ -x $BASE/bon-env/bin/python ]] || "$UV" venv --python 3.12 "$BASE/bon-env"
"$UV" pip install --python "$BASE/bon-env/bin/python" ${UV_ARGS:-} \
    "${VLLM_SPEC:-vllm}" transformers accelerate datasets math-verify numpy scipy matplotlib
"$BASE/bon-env/bin/python" -c "import vllm, torch, transformers; print('OK vllm', vllm.__version__, 'torch', torch.__version__)"
