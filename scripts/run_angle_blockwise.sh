#!/usr/bin/env bash

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

DATASET="taco"

if [[ $# -gt 0 ]]; then
    case "$1" in
        taco|arctic)
            DATASET="$1"
            shift
            ;;
    esac
fi

BLOCKS=("$@")
if [[ ${#BLOCKS[@]} -eq 0 ]]; then
    BLOCKS=(0 2 4 6)
fi

echo "════════════════════════════════════════════════════════════════"
echo "  [Angle Block Pretraining] 시작 (${TIMESTAMP})"
echo "  로그 디렉토리: ${LOG_DIR}"
echo "  dataset: ${DATASET}"
echo "  blocks: ${BLOCKS[*]}"
echo "  XFORMERS_DISABLED=TRUE"
echo "════════════════════════════════════════════════════════════════"

TOTAL=${#BLOCKS[@]}
IDX=1

for block in "${BLOCKS[@]}"; do
    EXP_NAME="brainco/ours_vectors/block/dinov2_pretraining_${DATASET}_block${block}"
    YAML_PATH="config/experiment/${EXP_NAME}.yaml"
    LOG_FILE="${LOG_DIR}/${DATASET}_block${block}.log"

    if [[ ! -f "${YAML_PATH}" ]]; then
        echo "[${IDX}/${TOTAL}] block ${block} 스킵: ${YAML_PATH} 없음" | tee "${LOG_FILE}"
        IDX=$((IDX + 1))
        continue
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    printf "  [%d/%d] block %s  (started %s)\n" "${IDX}" "${TOTAL}" "${block}" "$(date +%H:%M:%S)"
    echo "  experiment: ${EXP_NAME}"
    echo "  log: ${LOG_FILE}"
    echo "════════════════════════════════════════════════════════════════"

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 \
        python train.py "+experiment=${EXP_NAME}" \
        2>&1 | tee "${LOG_FILE}"; then
        echo "[✓] $(date +%H:%M:%S) — block ${block} 완료"
    else
        echo "[✗] $(date +%H:%M:%S) — block ${block} 실패"
        echo "EXPERIMENT_FAILED" >> "${LOG_FILE}"
    fi

    IDX=$((IDX + 1))
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  모든 실행 종료"
echo "  로그 디렉토리: ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════════"
