# Pytorch no longer provides conda packages
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# install xformers (make sure that the cuda versions are compatible)
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu130

uv pip install hydra-core hydra-colorlog wandb matplotlib einops tqdm scipy scikit-learn h5py rich seaborn scikit-learn moviepy
uv pip install lightning

# install huggingface datasets
uv pip install datasets safetensors fsspec requests pyyaml

uv pip install sympy

uv pip install rootutils opencv-python pytorch-kinematics gdown pre-commit
uv pip install mcap-ros2-support mcap joblib
uv pip install pandas tqdm