#!/bin/bash
pip install uv
uv venv --python 3.10.15
source .venv/bin/activate

# Data Generation installation
uv pip install --upgrade pip
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
cd dependencies/IsaacLab
./isaaclab.sh --install
uv pip install viser
uv pip install viser[examples]
uv pip install -U "jax[cuda12]"==0.5.3
cd ..
uv pip install -e trajgen
cd ..
uv pip install -e .