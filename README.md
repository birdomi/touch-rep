# BrainCo Tactile SSL

This branch keeps the BrainCo tactile representation and downstream task code paths.
Legacy D360, Touch-and-Go, Xela, cross-sensor, and unrelated dataset code/configs have
been removed.

## Installation

```bash
bash local_env.sh
pip install -e .
```

## Data Layout

Expected local data roots:

```text
dataset/brainco/
pretraining_dataset/brainco/
pretraining_dataset/vector_dataset/
```

`pretraining_dataset/vector_dataset/` is kept for the angle-vector pretraining
configs under `config/experiment/brainco/ours_vectors`.

## Training

BrainCo pretraining:

```bash
python train.py +experiment=brainco/uni_input/dinov2_pretraining.yaml
```

Angle/vector pretraining:

```bash
python train.py +experiment=brainco/ours_vectors/dinov2_pretraining_all.yaml
```

BrainCo downstream tasks:

```bash
python train_task_brainco.py +experiment=brainco/ours_vectors/task/grasp_prediction/dinov2_multi_combined.yaml
python train_task_brainco_vision.py +experiment=brainco/task/grasp_prediction/resnet18.yaml
```
