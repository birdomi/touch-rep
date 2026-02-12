# Downstream tasks
## Overview

We provide examples of how to use the Sparsh representations on downstream tasks. These examples focus on supervised learning tasks, including regression and classification. 

These tasks serve as a reference for adapting Sparsh touch encoders to different applications, such as policy learning. 

### Downstream tasks with Sparsh-X and Sparsh-skin

Please take a look to the task config in: 
- `config/experiment/xela/task/force/dinov2.yaml` for Xela tasks
- `config/experiment/d360/downstream_task/classification/dino.yaml` for D360 tasks 

and examples of data config at 

- `config/data/d360_classification_gs.yaml` for an object-material classification task with D360 Sparsh-X.
- `config/data/xela_force.yaml` for force regression task with Sparsh-skin

An example for launching a training job for the corresponding downstream task using pre-trained and frozen Sparsh-X encoder (with all touch sensing modalities, image-audio-imu-pressure) is:

```python
python train_task.py +experiment=d360/downstream_task/classification/dino.yaml paths=<CONFIG PATHS> paths.data_root=<PATH TO DATASET ROOT> +paths.encoder_checkpoint_root=<PATH TO Sparsh-X ENCODER CHECKPOINT> wandb=<WANDB GROUP> use_img=true use_mic=true use_imu=true use_pressure=true 
```

It is also possible to train the task end-to-end (E2E) by not loading the encoder checkpoint and allowing to update the encoder's weights:
```python
python train_task.py +experiment=d360/downstream_task/classification/e2e.yaml paths=<CONFIG PATHS> paths.data_root=<PATH TO DATASET ROOT> wandb=<WANDB GROUP> task.train_encoder=True trainer.save_probe_weights_only=False
```

Similarly you can also launch a training job for Sparsh-skin by appropriately changing the task config and data config files as follows: 

### Force regression

Another interesting task is to infer the amount of normal force experienced by the D360 elastomer when in contact with an object. We evaluate this in a lab setting, using a hemi-spherical probe to apply normal loading cycles on the sensor.

Please take a look to the task config in `config/experiment/d360/downstream_task/force/dino.yaml` and an example of data config `/home/carohiguera/fr/sparsh_release/tactile-ssl/config/data/d360_contact_gs.yaml` for a normal force regression task.

An example for launching a training job for this regression task is:

```python
python train_task.py +experiment=d360/downstream_task/force/dino.yaml paths=<CONFIG PATHS> paths.data_root=<PATH TO DATASET ROOT> +paths.encoder_checkpoint_root=<PATH TO Sparsh-X ENCODER CHECKPOINT> wandb=<WANDB GROUP> use_img=true use_mic=true use_imu=true use_pressure=true 
```