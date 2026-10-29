#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${CUDA_HOME:=$(cd ../cuda_env && pwd)}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"
PYTHON_BIN="${PYTHON_BIN:-../pynet_env/bin/python}"
if [[ -x ../pynet_env/bin/x86_64-conda-linux-gnu-g++ ]]; then
  export CXX="$(cd ../pynet_env/bin && pwd)/x86_64-conda-linux-gnu-g++"
  export CC="$(cd ../pynet_env/bin && pwd)/x86_64-conda-linux-gnu-gcc"
fi
export PATH="$(dirname "$(realpath "$PYTHON_BIN")"):$PATH"
"$PYTHON_BIN" "${PYNET_SCRIPT:-benchmark.py}" "$@"
