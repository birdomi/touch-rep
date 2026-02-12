#!/bin/bash
#SBATCH --requeue
#SBATCH --array=0-11
#SBATCH --job-name=xela_downstream_task
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account gum
#SBATCH --qos gum
#SBATCH --time=03-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-type=FAIL
#SBATCH --signal=SIGUSR1@90

TASK=$1
SENSOR=$2
PATHS=$3
BASE_MAX_EPOCHS=51
SSL_METHODS=("dinov2" "finetune" "e2e")
TRAIN_DATA_BUDGET=("1.0" "0.5" "0.33" "0.03")
MAX_EPOCHS=("25" "50" "75" "750")

for ssl_method in "${SSL_METHODS[@]}";
do
  for((i=0; i<${#TRAIN_DATA_BUDGET[@]}; i++));
  do
    FLAT_SSL_METHODS+=("$ssl_method")
    FLAT_TRAIN_DATA_BUDGET+=("${TRAIN_DATA_BUDGET[$i]}")
    FLAT_MAX_EPOCHS+=("${MAX_EPOCHS[$i]}")
  done
done
echo ${FLAT_SSL_METHODS[@]}
echo ${FLAT_TRAIN_DATA_BUDGET[@]}
echo ${FLAT_MAX_EPOCHS[@]}

source /data/home/$USER/miniforge3/etc/profile.d/conda.sh
conda activate tactile_ssl --no-stack
wandb enabled

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


echo srun python train_task.py \
    --config-name=experiment/xela/task/${TASK}/${FLAT_SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
    paths=$PATHS hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
    train_data_budget=${FLAT_TRAIN_DATA_BUDGET[$SLURM_ARRAY_TASK_ID]} trainer.max_epochs=${FLAT_MAX_EPOCHS[$SLURM_ARRAY_TASK_ID]} \
    paths.encoder_checkpoint_root=/fsx-gum/akashsharma02/percepskin_models +trainer.log_frequency=100

srun python train_task.py \
    --config-name=experiment/xela/task/${TASK}/${FLAT_SSL_METHODS[$SLURM_ARRAY_TASK_ID]} \
    paths=$PATHS hydra.job.id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} wandb=gum_rep_learning \
    train_data_budget=${FLAT_TRAIN_DATA_BUDGET[$SLURM_ARRAY_TASK_ID]} trainer.max_epochs=${FLAT_MAX_EPOCHS[$SLURM_ARRAY_TASK_ID]} \
    paths.encoder_checkpoint_root=/fsx-gum/akashsharma02/percepskin_models +trainer.log_frequency=100
