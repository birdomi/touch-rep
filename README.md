# Tactile Representation for Homogeneous Tactile Sensors.

This repository contains the implementation of the tactile representation for homogeneous tactile sensors. This repository is based on the [Sparsh-Skin](https://github.com/facebookresearch/sparsh-multisensory-touch) repository.

## Installation
```bash
https://github.com/birdomi/touch-rep.git
cd touch-rep

bash local_env.sh 
pip install -e .
```

## Getting Started
#### 1. Sparsh-Skin dataset
Download the dataset from [Hugging Face](https://huggingface.co/datasets/facebook/sparsh-skin-dataset)!

#### 2. BrainCo dataset
Currently, this is not available. We will provide it soon.

## Training Tactile Representation
#### 1. For xela sensor (sparsh-skin)
```bash
python train.py +experiment=xela/dinov2.yaml
```

#### 2. For brainco revo2 hand 
```bash
python train.py +experiment=brainco/dinov2.yaml
```