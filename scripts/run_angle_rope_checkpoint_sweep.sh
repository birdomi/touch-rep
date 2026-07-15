#!/usr/bin/env bash
# run_angle_rope_checkpoint_sweep.sh
# checkpoints/dinov2_angle_rope/checkpoints 안의 모든 .ckpt로
# AngleTransformer + BraincoAngleGraspDataset 기반 rope grasp prediction 실험을 순차 실행한다.
#
# 사용법:
#   bash scripts/run_angle_rope_checkpoint_sweep.sh
#   bash scripts/run_angle_rope_checkpoint_sweep.sh --dry-run
#
# 결과:
#   scripts/logs/<timestamp>/  - checkpoint별 전체 로그
#   scripts/results_angle_rope_ckpt_sweep_<timestamp>.txt  - 요약 테이블
#   scripts/results_angle_rope_ckpt_sweep_<timestamp>.csv  - CSV (spreadsheet용)

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

# -- 경로 설정 -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
SUMMARY_TXT="${SCRIPT_DIR}/results_angle_rope_ckpt_sweep_${TIMESTAMP}.txt"
SUMMARY_CSV="${SCRIPT_DIR}/results_angle_rope_ckpt_sweep_${TIMESTAMP}.csv"

EXPERIMENT="grasp_prediction_rope/dinov2_combined"
EXPERIMENT_PATH="brainco/ours_vectors/task/${EXPERIMENT}"
CKPT_DIR="checkpoints/dinov2_angle_rope/checkpoints"

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            sed -n '1,14p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] 알 수 없는 인자: $1" >&2
            echo "  사용법: $0 [--dry-run]" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_DIR}"

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] 체크포인트 디렉토리를 찾을 수 없습니다: ${CKPT_DIR}" >&2
    exit 1
fi

mapfile -t CHECKPOINTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[ERROR] 실행할 .ckpt 파일이 없습니다: ${CKPT_DIR}" >&2
    exit 1
fi

TOTAL=${#CHECKPOINTS[@]}

# -- 결과 추출 함수 -------------------------------------------------------------
extract_results() {
    local log_file="$1"
    python3 - "${log_file}" <<'PYEOF'
import sys, re

path = sys.argv[1]
try:
    text = open(path).read()
except OSError:
    print(",,,,,,,,NOFILE")
    sys.exit(0)

if "EXPERIMENT_FAILED" in text:
    print(",,,,,,,,FAILED")
    sys.exit(0)

def extract_epoch(label):
    pat = (
        rf'K-FOLD SUMMARY \({re.escape(label)}\).*?'
        r'Mean\s+([\d.]+|nan)\s+([\d.]+|nan).*?'
        r'Std\s+([\d.]+|nan)\s+([\d.]+|nan)'
    )
    m = re.search(pat, text, re.DOTALL)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return ('', '', '', '')

last = extract_epoch('Last Epoch')
best = extract_epoch('Best Epoch')

if any(last) or any(best):
    print(','.join([*last, *best, 'OK']))
else:
    def extract_single(label):
        pat = rf'{label}\s+[^A-Za-z0-9]*\s+Acc:\s+([\d.]+|nan)\s+F1:\s+([\d.]+|nan)'
        m = re.search(pat, text)
        if m:
            return m.group(1), m.group(2), '0.0000', '0.0000'
        return ('', '', '', '')

    last = extract_single('Last')
    best = extract_single('Best')
    if any(last) or any(best):
        print(','.join([*last, *best, 'OK']))
    else:
        print(",,,,,,,,INCOMPLETE")
PYEOF
}

# -- 실행 함수 -----------------------------------------------------------------
run_checkpoint() {
    local ckpt_path="$1"
    local idx="$2"

    local ckpt_name
    ckpt_name="$(basename "${ckpt_path}" .ckpt)"
    local log_file="${LOG_DIR}/grasp_prediction_rope__dinov2_combined__${ckpt_name}.log"

    echo ""
    echo "================================================================"
    printf "  [%d/%d]  %s  (started %s)\n" "${idx}" "${TOTAL}" "${ckpt_path}" "$(date +%H:%M:%S)"
    echo "================================================================"

    if [[ "${DRY_RUN}" == true ]]; then
        printf "[DRY-RUN] XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 python train_task_brainco_angle.py '+experiment=%s' task.checkpoint_encoder=%s --all_split\n" \
            "${EXPERIMENT_PATH}" "${ckpt_path}" | tee "${log_file}"
        return 0
    fi

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 \
        python train_task_brainco_angle.py \
        "+experiment=${EXPERIMENT_PATH}" \
        "task.checkpoint_encoder=${ckpt_path}" \
        --all_split \
        2>&1 | tee "${log_file}"; then
        echo ""
        echo "[OK] $(date +%H:%M:%S) - ${ckpt_path} 완료"
    else
        echo ""
        echo "[FAIL] $(date +%H:%M:%S) - ${ckpt_path} 실패"
        echo "EXPERIMENT_FAILED" >> "${log_file}"
    fi
}

echo "================================================================"
echo "  [Angle Rope Checkpoint Sweep] 실험 시작: ${TOTAL}개 (${TIMESTAMP})"
echo "  experiment : ${EXPERIMENT_PATH}"
echo "  ckpt dir   : ${CKPT_DIR}"
echo "  log dir    : ${LOG_DIR}"
echo "================================================================"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  - ${ckpt}"
done

# -- 실험 순차 실행 -------------------------------------------------------------
idx=1
for ckpt in "${CHECKPOINTS[@]}"; do
    run_checkpoint "${ckpt}" "${idx}"
    idx=$((idx + 1))
done

# -- 결과 취합 -----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  결과 취합 중..."
echo "================================================================"

echo "experiment,checkpoint,last_mean_acc,last_mean_f1,last_std_acc,last_std_f1,best_mean_acc,best_mean_f1,best_std_acc,best_std_f1,status" \
    > "${SUMMARY_CSV}"

HEADER=$(printf "%-35s  %-18s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "Experiment" "Checkpoint" "Epoch" "MeanAcc" "MeanF1" "StdAcc" "StdF1" "Status")
SEP=$(printf "%s" \
    "-----------------------------------  ------------------  -----  ----------  ----------  ----------  ----------  ------")

{
    echo ""
    echo "=========================================================================================="
    printf "  ANGLE ROPE CHECKPOINT SWEEP RESULTS  (%s)\n" "${TIMESTAMP}"
    echo "=========================================================================================="
    echo ""
    echo "  Experiment: ${EXPERIMENT_PATH}"
    echo "  Checkpoint dir: ${CKPT_DIR}"
    echo ""
    echo "${HEADER}"
    echo "${SEP}"
} | tee "${SUMMARY_TXT}"

for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "${ckpt}" .ckpt)"
    log_file="${LOG_DIR}/grasp_prediction_rope__dinov2_combined__${ckpt_name}.log"

    result=$(extract_results "${log_file}")
    IFS=',' read -r lm_acc lm_f1 ls_acc ls_f1 bm_acc bm_f1 bs_acc bs_f1 status <<< "${result}"

    printf "%-35s  %-18s  %-5s  %-10s  %-10s  %-10s  %-10s  %s\n" \
        "dinov2_combined" "${ckpt_name}" "Last" \
        "${lm_acc:-n/a}" "${lm_f1:-n/a}" "${ls_acc:-n/a}" "${ls_f1:-n/a}" \
        "${status:-UNKNOWN}" \
        | tee -a "${SUMMARY_TXT}"

    printf "%-35s  %-18s  %-5s  %-10s  %-10s  %-10s  %-10s\n" \
        "" "" "Best" \
        "${bm_acc:-n/a}" "${bm_f1:-n/a}" "${bs_acc:-n/a}" "${bs_f1:-n/a}" \
        | tee -a "${SUMMARY_TXT}"

    echo "${EXPERIMENT_PATH},${ckpt},${lm_acc:-},${lm_f1:-},${ls_acc:-},${ls_f1:-},${bm_acc:-},${bm_f1:-},${bs_acc:-},${bs_f1:-},${status:-UNKNOWN}" \
        >> "${SUMMARY_CSV}"
done

{
    echo ""
    echo "${SEP}"
    echo ""
    echo "========================================================================"
    echo "  Logs   : ${LOG_DIR}/"
    echo "  Summary: ${SUMMARY_TXT}"
    echo "  CSV    : ${SUMMARY_CSV}"
    echo "========================================================================"
} | tee -a "${SUMMARY_TXT}"
