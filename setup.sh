#!/usr/bin/env bash
set -e

# ===== 0) Conda environment =====
ENV_NAME="medspa"
PY=3.10

if ! conda env list | grep -q "^${ENV_NAME}\s"; then
  conda create -n ${ENV_NAME} python=${PY} -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# ===== 1) PyTorch + CUDA =====
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ===== 2) Install project code in editable mode =====
cd src/r1-v
pip install -e ".[dev]"
cd ../../

# ===== 3) Install additional dependencies =====
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# ===== 4) vLLM support =====
pip install vllm==0.7.2

# ===== 5) Pin Transformers to a specific commit =====
pip install --no-deps --force-reinstall \
  git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

echo "✅ Setup completed. Environment: ${ENV_NAME}"

python -c "import torch, transformers; print('CUDA:', torch.cuda.is_available(), 'Torch:', torch.__version__, 'Transformers:', transformers.__version__)"