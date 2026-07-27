#!/usr/bin/env bash
# Run temporal-3 DINOv2 pretrained/scratch grasp-prediction experiments.
#
# The delegated runner evaluates both protocols by default:
#   - ID:  all-object episode-level K-fold evaluation
#   - OOD: five-way leave-one-object-out evaluation
#
# Usage:
#   bash scripts/run_xyz_grasp_prediction_temp3_id_ood.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_grasp_prediction_temp3_id_ood.sh
#   RUN_ID=0 bash scripts/run_xyz_grasp_prediction_temp3_id_ood.sh
#   RUN_OOD=0 bash scripts/run_xyz_grasp_prediction_temp3_id_ood.sh
#   RUN_SCRATCH=0 bash scripts/run_xyz_grasp_prediction_temp3_id_ood.sh
#   bash scripts/run_xyz_grasp_prediction_temp3_id_ood.sh -- trainer.max_epochs=50

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT:-brainco/ours_3d/task/grasp_prediction/dinov2_all_rope_temp3}"
SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT:-brainco/ours_3d/task/grasp_prediction/dinov2_all_rope_temp3_scratch}"
RUN_OOD="${RUN_OOD:-${RUN_OBJECTWISE:-1}}"

if [[ "${RUN_OOD}" != "0" && "${RUN_OOD}" != "1" ]]; then
    echo "[ERROR] RUN_OOD must be 0 or 1." >&2
    exit 1
fi

PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT}" \
    SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT}" \
    RUN_OBJECTWISE="${RUN_OOD}" \
    exec bash "${SCRIPT_DIR}/run_xyz_grasp_prediction_multiseed.sh" "$@"
