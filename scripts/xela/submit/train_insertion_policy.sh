#!/bin/bash
#SBATCH --requeue
#SBATCH --array=0
#SBATCH --job-name=xela_insertion_policy
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account gum
#SBATCH --qos gum_high
#SBATCH --time=36:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-type=FAIL
#SBATCH --signal=SIGUSR1@90

BASE_MAX_EPOCHS=1000
SSL_METHODS=("dinov2_act" "finetune_act" "e2e_act")

source /data/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate tactile_ssl --no-stack
wandb enabled

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo srun python train_task.py \
    --config-name=experiment/xela/task/plug_insertion_policy/${SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
    paths=fair-aws hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
    trainer.max_epochs=1000 \
    paths.encoder_checkpoint_root=/fsx-gum/akashsharma02/percepskin_models +trainer.log_frequency=100

srun python train_task.py \
    --config-name=experiment/xela/task/plug_insertion_policy/${SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
    paths=fair-aws hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
    trainer.max_epochs=1000 \
    paths.encoder_checkpoint_root=/fsx-gum/akashsharma02/percepskin_models +trainer.log_frequency=100 experiment_name=vision_only_bugfix
