#!/bin/bash
# UltraQuery official evaluation environment (Blackwell-compatible)
set -e
export PIP_CACHE_DIR=/mnt/ssd/pipcache
mkdir -p /mnt/ssd/envs /mnt/ssd/pipcache
PYENV=/mnt/ssd/envs/ultra/bin

echo "[$(date +%H:%M:%S)] creating conda env (python 3.10) ..."
/root/miniconda3/bin/conda create -y --prefix /mnt/ssd/envs/ultra python=3.10

echo "[$(date +%H:%M:%S)] installing torch 2.8.0+cu128 ..."
$PYENV/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "[$(date +%H:%M:%S)] installing torch-scatter + PyG ..."
$PYENV/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
$PYENV/pip install torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html || echo "torch-sparse skipped"
$PYENV/pip install torch-geometric==2.4.0 ninja easydict pyyaml

echo "[$(date +%H:%M:%S)] verifying GPU + scatter ..."
$PYENV/python -c "
import torch, torch_scatter
x = torch.randn(100, device='cuda:0')
idx = torch.randint(0, 10, (100,), device='cuda:0')
out = torch_scatter.scatter_add(x, idx, dim=0)
print('GPU scatter OK:', out.shape, torch.cuda.get_device_name(0))
import torch_geometric; print('PyG', torch_geometric.__version__)
"
echo "[$(date +%H:%M:%S)] ENV SETUP DONE"
