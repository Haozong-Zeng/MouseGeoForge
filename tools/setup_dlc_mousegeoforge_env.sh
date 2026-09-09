#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLC_DIR="${ROOT_DIR}/DeepLabCut"
VENV_DIR="${DLC_DIR}/.venv-mousegeoforge"
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x /snap/bin/uv ]]; then
    UV_BIN=/snap/bin/uv
fi
if [[ -z "${UV_BIN}" ]]; then
    echo "uv was not found." >&2
    exit 1
fi

"${UV_BIN}" venv --python 3.11 "${VENV_DIR}"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
    -e "${DLC_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import deeplabcut
import torch

print(f"deeplabcut={deeplabcut.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
