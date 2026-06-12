#!/usr/bin/env bash
# run_angle_blockwise_experiments.sh
# checkpoints/dinov2_angle_blockwise 의 blockwise pretrained encoder 4개로
# AngleTransformer grasp prediction 실험을 순차 실행하고 결과를 요약한다.
#
# 사용법:
#   bash scripts/run_angle_blockwise_experiments.sh
#   bash scripts/run_angle_blockwise_experiments.sh -m "epoch-0200-0,epoch-0200-2"
#   bash scripts/run_angle_blockwise_experiments.sh -m "checkpoints/dinov2_angle_blockwise/epoch-0200-0.ckpt"
#
# 결과:
#   scripts/logs/<timestamp>/  — 실험별 전체 로그
#   scripts/results_angle_blockwise_<timestamp>.txt  — 요약 테이블
#   scripts/results_angle_blockwise_<timestamp>.csv  — CSV (spreadsheet용)

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
SUMMARY_TXT="${SCRIPT_DIR}/results_angle_blockwise_${TIMESTAMP}.txt"
SUMMARY_CSV="${SCRIPT_DIR}/results_angle_blockwise_${TIMESTAMP}.csv"

mkdir -p "${LOG_DIR}"

# ── 실험 목록 ─────────────────────────────────────────────────────────────────
PRETRAINED_EXPERIMENTS=(
    "grasp_prediction/dinov2_multi"
    "grasp_prediction/dinov2_multi_freeze"
    "grasp_prediction/dinov2_multi_mask"
    "grasp_prediction/dinov2_multi_mask_freeze"
)

BLOCKWISE_MODELS=(
    "epoch-0200-0"
    "epoch-0200-2"
    "epoch-0200-4"
    "epoch-0200-6"
)

CHECKPOINT_DIR="checkpoints/dinov2_angle_blockwise"

# ── CLI 인자 파싱 ─────────────────────────────────────────────────────────────
# 사용 가능한 옵션:
#   -m|--models <m1,m2,...>    blockwise 체크포인트 목록
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--models)
            shift
            IFS=',' read -ra SELECTED_MODELS <<< "$1"
            shift
            ;;
        *)
            echo "[ERROR] 알 수 없는 인자: $1" >&2
            echo "  사용법: $0 [-m model1,model2,...]" >&2
            exit 1
            ;;
    esac
done

[[ ${#SELECTED_MODELS[@]} -eq 0 ]] && SELECTED_MODELS=("${BLOCKWISE_MODELS[@]}")

# ── ALL_EXPERIMENTS 동적 생성 ────────────────────────────────────────────────
ALL_EXPERIMENTS=()
for model in "${SELECTED_MODELS[@]}"; do
    for exp in "${PRETRAINED_EXPERIMENTS[@]}"; do
        ALL_EXPERIMENTS+=("${exp}:${model}")
    done
done

TOTAL=${#ALL_EXPERIMENTS[@]}

# ── 엔트리 파싱 헬퍼 ──────────────────────────────────────────────────────────
# 형식: "task/exp_name:model_spec"
parse_entry() {
    local entry="$1"
    PARSED_EXP="${entry%%:*}"
    PARSED_CKPT="${entry#*:}"
}

# ── 체크포인트 경로 해석 ──────────────────────────────────────────────────────
# spec이 경로(/ 포함) 또는 .ckpt 이면 그대로 사용
# spec이 스템이면 checkpoints/dinov2_angle_blockwise/<spec>.ckpt 로 해석
resolve_ckpt() {
    local spec="$1"

    if [[ "${spec}" == *"/"* || "${spec}" == *.ckpt ]]; then
        echo "${spec}"
        return
    fi

    echo "${CHECKPOINT_DIR}/${spec}.ckpt"
}

# ── 결과 추출 함수 ────────────────────────────────────────────────────────────
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
    print(",,,,,,,,INCOMPLETE")
PYEOF
}

# ── 실행 함수 ─────────────────────────────────────────────────────────────────
run_experiment() {
    local entry="$1"
    local idx="$2"

    parse_entry "${entry}"
    local exp="${PARSED_EXP}"
    local ckpt_spec="${PARSED_CKPT}"

    local task_dir="${exp%%/*}"
    local exp_name="${exp##*/}"
    local exp_path="brainco/ours_vectors/task/${exp}"

    local resolved_ckpt
    resolved_ckpt=$(resolve_ckpt "${ckpt_spec}")

    if [[ ! -f "${resolved_ckpt}" ]]; then
        echo "[ERROR] checkpoint 없음: ${resolved_ckpt}" >&2
        return 1
    fi

    local model_tag
    model_tag="$(basename "${resolved_ckpt}" .ckpt | tr '/' '_')"
    local log_file="${LOG_DIR}/${task_dir}__${exp_name}__${model_tag}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    printf "  [%d/%d]  %s  (started %s)\n" "${idx}" "${TOTAL}" "${exp}" "$(date +%H:%M:%S)"
    printf "  blockwise checkpoint: %s\n" "${resolved_ckpt}"
    echo "════════════════════════════════════════════════════════════════"

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 \
        python train_task_brainco_angle.py \
        "+experiment=${exp_path}" \
        "task.checkpoint_encoder=${resolved_ckpt}" \
        --all_split \
        2>&1 | tee "${log_file}"; then
        echo ""
        echo "[✓] $(date +%H:%M:%S) — ${exp} 완료"
    else
        echo ""
        echo "[✗] $(date +%H:%M:%S) — ${exp} 실패"
        echo "EXPERIMENT_FAILED" >> "${log_file}"
    fi
}

# ── 선택된 모델 목록 출력 ─────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  [Angle Blockwise Grasp] 실험 시작: ${TOTAL}개  (${TIMESTAMP})"
echo "  로그 디렉토리: ${LOG_DIR}"
echo "  blockwise checkpoint(s):"
for m in "${SELECTED_MODELS[@]}"; do
    echo "    - $(resolve_ckpt "${m}")"
done
echo "════════════════════════════════════════════════════════════════"

# ── 실험 순차 실행 ────────────────────────────────────────────────────────────
idx=1
for entry in "${ALL_EXPERIMENTS[@]}"; do
    run_experiment "${entry}" "${idx}"
    idx=$((idx + 1))
done

# ── 결과 취합 ─────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  결과 취합 중..."
echo "════════════════════════════════════════════════════════════════"

echo "task,experiment,blockwise_checkpoint,last_mean_acc,last_std_acc,last_mean_f1,last_std_f1,best_mean_acc,best_std_acc,best_mean_f1,best_std_f1,status" \
    > "${SUMMARY_CSV}"

HEADER=$(printf "%-25s  %-35s  %-28s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "Task" "Experiment" "Checkpoint" "Epoch" "MeanAcc" "StdAcc" "MeanF1" "StdF1" "Status")
SEP=$(printf "%s" \
    "-------------------------  -----------------------------------  ----------------------------  -----  ----------  ----------  ----------  ----------  ------")

{
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    printf "  ANGLE BLOCKWISE GRASP PREDICTION RESULTS  (%s)\n" "${TIMESTAMP}"
    echo "════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "${HEADER}"
    echo "${SEP}"
} | tee "${SUMMARY_TXT}"

prev_model=""
for entry in "${ALL_EXPERIMENTS[@]}"; do
    parse_entry "${entry}"
    exp="${PARSED_EXP}"
    ckpt_spec="${PARSED_CKPT}"

    task_dir="${exp%%/*}"
    exp_name="${exp##*/}"
    resolved_ckpt=$(resolve_ckpt "${ckpt_spec}")
    model_tag="$(basename "${resolved_ckpt}" .ckpt | tr '/' '_')"
    log_file="${LOG_DIR}/${task_dir}__${exp_name}__${model_tag}.log"

    result=$(extract_results "${log_file}")
    IFS=',' read -r lm_acc ls_acc lm_f1 ls_f1 bm_acc bs_acc bm_f1 bs_f1 status <<< "${result}"

    if [[ "${resolved_ckpt}" != "${prev_model}" ]]; then
        {
            echo ""
            printf "  [Blockwise checkpoint: %s]\n" "${resolved_ckpt}"
        } | tee -a "${SUMMARY_TXT}"
        prev_model="${resolved_ckpt}"
    fi

    printf "%-25s  %-35s  %-28s  %-5s  %-10s  %-10s  %-10s  %-10s  %s\n" \
        "${task_dir}" "${exp_name}" "${model_tag}" "Last" \
        "${lm_acc:-n/a}" "${ls_acc:-n/a}" "${lm_f1:-n/a}" "${ls_f1:-n/a}" \
        "${status:-UNKNOWN}" \
        | tee -a "${SUMMARY_TXT}"

    printf "%-25s  %-35s  %-28s  %-5s  %-10s  %-10s  %-10s  %-10s\n" \
        "" "" "" "Best" \
        "${bm_acc:-n/a}" "${bs_acc:-n/a}" "${bm_f1:-n/a}" "${bs_f1:-n/a}" \
        | tee -a "${SUMMARY_TXT}"

    echo "${task_dir},${exp_name},${resolved_ckpt},${lm_acc:-},${ls_acc:-},${lm_f1:-},${ls_f1:-},${bm_acc:-},${bs_acc:-},${bm_f1:-},${bs_f1:-},${status:-UNKNOWN}" \
        >> "${SUMMARY_CSV}"
done

{
    echo ""
    echo "${SEP}"
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Logs   : ${LOG_DIR}/"
    echo "  Summary: ${SUMMARY_TXT}"
    echo "  CSV    : ${SUMMARY_CSV}"
    echo "════════════════════════════════════════════════════════════════════════════════"
} | tee -a "${SUMMARY_TXT}"
