#!/bin/bash
#SBATCH --requeue
#SBATCH --job-name=percepskin_mae
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --account=gum
#SBATCH --qos=gum_high
#SBATCH --time=03-00:00:00
#SBATCH --signal=SIGUSR1@90

source /data/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate tactile_ssl --no-stack
wandb enabled

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun python train.py +experiment=$1 paths=fair-aws wandb=gum_rep_learning hydra.job.id=$SLURM_JOB_ID $2 $3 $4 $5 $6 $7 $8 $9
