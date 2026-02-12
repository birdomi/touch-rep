#!/bin/bash -e
shopt -s expand_aliases
if type -P micromamba; then
	echo "micromamba detect, using micromamba inplace of mamba"
	alias conda=micromamba
	eval "$(micromamba shell hook --shell bash)"
elif type -P mamba || type -P conda; then
	eval "$(conda shell.bash hook)"
else
	echo please install mamba or micromamba
	exit
fi


mamba create -y --name tactile_ssl python=3.10

conda activate tactile_ssl

# Pytorch no longer provides conda packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# install xformers (make sure that the cuda versions are compatible)
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126

pip install hydra-core hydra-colorlog wandb matplotlib einops tqdm scipy scikit-learn h5py rich seaborn scikit-learn moviepy
pip install lightning

# install huggingface datasets
pip install datasets safetensors fsspec requests pyyaml

pip install sympy

pip install rootutils opencv-python pytorch-kinematics gdown pre-commit
pip install mcap-ros2-support mcap joblib
pip install pandas tqdm