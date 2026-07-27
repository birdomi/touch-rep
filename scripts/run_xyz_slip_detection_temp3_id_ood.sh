#!/usr/bin/env bash
# Compare temporal-3 DINOv2 pretrained and scratch models on XYZ slip detection.
#
# Protocols:
#   - ID:  all-object episode-level K-fold evaluation
#   - OOD: leave-one-object-out evaluation over the five move objects
#
# Usage:
#   bash scripts/run_xyz_slip_detection_temp3_id_ood.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_slip_detection_temp3_id_ood.sh
#   RUN_ID=0 bash scripts/run_xyz_slip_detection_temp3_id_ood.sh
#   RUN_OOD=0 bash scripts/run_xyz_slip_detection_temp3_id_ood.sh
#   RUN_SCRATCH=0 bash scripts/run_xyz_slip_detection_temp3_id_ood.sh
#   bash scripts/run_xyz_slip_detection_temp3_id_ood.sh -- trainer.max_epochs=50
#
# Optional environment variables:
#   SEEDS, SPLIT_SEED, NUM_FOLDS, OBJECTS
#   RUN_ID, RUN_OOD, RUN_PRETRAINED, RUN_SCRATCH
#   PRETRAINED_EXPERIMENT, SCRATCH_EXPERIMENT

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT:-brainco/ours_3d/task/slip_detection/dinov2_all_rope_temp3}"
SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT:-brainco/ours_3d/task/slip_detection/dinov2_all_rope_temp3_scratch}"
SEEDS="${SEEDS:-0 1 2}"
SPLIT_SEED="${SPLIT_SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-4}"
OBJECTS="${OBJECTS:-slip_box_move slip_case_move slip_doll_move slip_plastic_move slip_pot_move}"
RUN_ID="${RUN_ID:-1}"
RUN_OOD="${RUN_OOD:-1}"
RUN_PRETRAINED="${RUN_PRETRAINED:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"

for flag_name in RUN_ID RUN_OOD RUN_PRETRAINED RUN_SCRATCH; do
    flag_value="${!flag_name}"
    if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
        echo "[ERROR] ${flag_name} must be 0 or 1." >&2
        exit 1
    fi
done
if [[ "${RUN_ID}" != "1" && "${RUN_OOD}" != "1" ]]; then
    echo "[ERROR] Enable RUN_ID=1 and/or RUN_OOD=1." >&2
    exit 1
fi
if [[ "${RUN_PRETRAINED}" != "1" && "${RUN_SCRATCH}" != "1" ]]; then
    echo "[ERROR] Enable RUN_PRETRAINED=1 and/or RUN_SCRATCH=1." >&2
    exit 1
fi

echo "Temporal-3 XYZ slip-detection experiments"
echo "  pretrained: ${PRETRAINED_EXPERIMENT} (enabled: ${RUN_PRETRAINED})"
echo "  scratch:    ${SCRATCH_EXPERIMENT} (enabled: ${RUN_SCRATCH})"
echo "  ID:         ${RUN_ID} (${NUM_FOLDS} folds, split seed ${SPLIT_SEED})"
echo "  OOD:        ${RUN_OOD} (objects: ${OBJECTS})"
echo "  seeds:      ${SEEDS}"
echo

status=0

if [[ "${RUN_ID}" == "1" ]]; then
    echo "================================================================"
    echo "  Starting ID experiments"
    echo "================================================================"
    if ! PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT}" \
        SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT}" \
        SEEDS="${SEEDS}" \
        SPLIT_SEED="${SPLIT_SEED}" \
        NUM_FOLDS="${NUM_FOLDS}" \
        RUN_ALL_SPLITS=1 \
        RUN_PRETRAINED="${RUN_PRETRAINED}" \
        RUN_SCRATCH="${RUN_SCRATCH}" \
        bash "${SCRIPT_DIR}/run_xyz_slip_detection_id_multiseed.sh" "$@"; then
        echo "[ERROR] ID experiment runner failed." >&2
        status=1
    fi
fi

if [[ "${RUN_OOD}" == "1" ]]; then
    echo "================================================================"
    echo "  Starting OOD leave-one-object-out experiments"
    echo "================================================================"
    if ! PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT}" \
        SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT}" \
        SEEDS="${SEEDS}" \
        OBJECTS="${OBJECTS}" \
        RUN_PRETRAINED="${RUN_PRETRAINED}" \
        RUN_SCRATCH="${RUN_SCRATCH}" \
        bash "${SCRIPT_DIR}/run_xyz_slip_detection_objectwise_multiseed.sh" "$@"; then
        echo "[ERROR] OOD experiment runner failed." >&2
        status=1
    fi
fi

exit "${status}"
