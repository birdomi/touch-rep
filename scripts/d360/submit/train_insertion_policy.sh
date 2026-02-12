#!/bin/bash
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --job-name=d360_insertion_policy
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --account gum
#SBATCH --qos gum
#SBATCH --time=36:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-type=FAIL
#SBATCH --signal=SIGUSR1@90

SSL_METHODS=("dino" "dino_finetune" "e2e")

source /data/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate tactile_ssl --no-stack
wandb enabled

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# echo srun python train_task.py \
#     +experiment=d360/downstream_task/plug_insertion/${SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
#     paths=fair-aws hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
#     paths.encoder_checkpoint_root=/fsx-gum/shared/d360_mmt_models/20250226 use_img=true use_mic=false use_imu=false use_pressure=false fusion_type=vanilla

# srun python train_task.py \
#     +experiment=d360/downstream_task/plug_insertion/${SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
#     paths=fair-aws hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
#     paths.encoder_checkpoint_root=/fsx-gum/shared/d360_mmt_models/20250226 use_img=true use_mic=false use_imu=false use_pressure=false fusion_type=vanilla


echo srun python train_task.py \
    +experiment=d360/downstream_task/plug_insertion/${SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
    paths=fair-aws hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
    paths.encoder_checkpoint_root=/fsx-gum/shared/d360_mmt_models/20241219 use_img=true use_mic=false use_imu=false use_pressure=false fusion_type=vanilla

srun python train_task.py \
    +experiment=d360/downstream_task/plug_insertion/${SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
    paths=fair-aws hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
    paths.encoder_checkpoint_root=/fsx-gum/shared/d360_mmt_models/20241219 use_img=true use_mic=false use_imu=false use_pressure=false fusion_type=vanilla
