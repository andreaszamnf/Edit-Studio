#!/usr/bin/env bash
# Edit Studio installer for macOS and Linux.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Python 3.12"
uv venv .venv --python 3.12 --allow-existing
PY=.venv/bin/python

echo "==> PyTorch"
uv pip install --python "$PY" torch torchvision --torch-backend=auto

extras="bg"
if [ "$(uname)" = "Darwin" ]; then
  extras="bg,desktop"
elif command -v nvidia-smi >/dev/null 2>&1; then
  extras="bg,quant"
fi
echo "==> Edit Studio [$extras]"
uv pip install --python "$PY" -e ".[$extras]" --torch-backend=auto

"$PY" -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
echo
echo "Done. Start with ./run.sh"
