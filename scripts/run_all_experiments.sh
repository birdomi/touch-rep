#!/usr/bin/env bash
# run_all_experiments.sh
# 모든 grasp task 실험을 순차 실행하고 K-Fold 결과를 취합해 저장한다.
#
# 사용법:
#   bash scripts/run_all_experiments.sh              # 전체 실행
#   bash scripts/run_all_experiments.sh detection    # grasp_detection 만
#   bash scripts/run_all_experiments.sh prediction   # grasp_prediction 만
#
# 결과:
#   scripts/logs/<timestamp>/  — 실험별 전체 로그
#   scripts/results_<timestamp>.txt  — 요약 테이블
#   scripts/results_<timestamp>.csv  — CSV (spreadsheet용)

# sh로 실행 시 bash로 재실행
[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
SUMMARY_TXT="${SCRIPT_DIR}/results_${TIMESTAMP}.txt"
SUMMARY_CSV="${SCRIPT_DIR}/results_${TIMESTAMP}.csv"

mkdir -p "${LOG_DIR}"

# ── 실험 목록 ─────────────────────────────────────────────────────────────────
# 형식 1: "task_subdir/experiment_yaml_stem"
#          → yaml 파일의 checkpoint_encoder 그대로 사용
# 형식 2: "task_subdir/experiment_yaml_stem:epoch-0300-arctic"
#          → 파일명 스템만 쓰면 yaml의 체크포인트 디렉토리에서 자동 완성
# 형식 3: "task_subdir/experiment_yaml_stem:checkpoints/other_dir/epoch-0300.ckpt"
#          → 경로 구분자(/) 또는 .ckpt 확장자가 있으면 그대로 사용
# 주석 처리하면 해당 실험 스킵

DETECTION_EXPERIMENTS=(
    # "grasp_detection/dinov2_multi"
    # "grasp_detection/dinov2_multi_mask"
    # "grasp_detection/dinov2_multi_rm"
    # "grasp_detection/dinov2_multi_mask_r2h"
    # "grasp_detection/dinov2_from_scratch"
    # "grasp_detection/dinov2_multi_scratch"
    # "grasp_detection/dinov2"          # 단일 센서 기반
    # "grasp_detection/dinov2_cat"      # CAT task (별도 모듈)
)

PREDICTION_EXPERIMENTS=(
    "grasp_prediction/dinov2_multi_scratch"
    "grasp_prediction/dinov2_multi"
    "grasp_prediction/dinov2_multi_mask"
    "grasp_prediction/dinov2_multi_mask_rm"
    "grasp_prediction/dinov2_multi_rm"
    "grasp_prediction/dinov2_multi_r2h"
    "grasp_prediction/dinov2_multi_mask_r2h"
    "grasp_prediction/dinov2_from_scratch"
    # "grasp_prediction/dinov2"
    # "grasp_prediction/dinov2_cat"
)

# CLI 인자로 범위 지정 (없으면 전체)
MODE="${1:-all}"
case "${MODE}" in
    detection)  ALL_EXPERIMENTS=("${DETECTION_EXPERIMENTS[@]}") ;;
    prediction) ALL_EXPERIMENTS=("${PREDICTION_EXPERIMENTS[@]}") ;;
    *)          ALL_EXPERIMENTS=("${DETECTION_EXPERIMENTS[@]}" "${PREDICTION_EXPERIMENTS[@]}") ;;
esac

TOTAL=${#ALL_EXPERIMENTS[@]}

# ── 엔트리 파싱 헬퍼 ─────────────────────────────────────────────────────────
# "task/exp[:ckpt_spec]" 형식에서 exp와 ckpt_override를 전역 변수로 반환한다.
parse_entry() {
    local entry="$1"
    if [[ "${entry}" == *":"* ]]; then
        PARSED_EXP="${entry%%:*}"
        PARSED_CKPT="${entry##*:}"
    else
        PARSED_EXP="${entry}"
        PARSED_CKPT=""
    fi
}

# ── 체크포인트 경로 해석 헬퍼 ────────────────────────────────────────────────
# spec이 스템(슬래시·확장자 없음)이면 yaml의 checkpoint_encoder 디렉토리를 읽어
# "checkpoints/<dir>/<stem>.ckpt" 로 완성한다.
# 슬래시 포함 또는 .ckpt로 끝나면 그대로 반환. 비어 있으면 빈 문자열 반환.
resolve_ckpt() {
    local exp="$1"
    local spec="$2"

    [[ -z "${spec}" ]] && echo "" && return

    # 명시적 경로(/ 포함) 또는 파일명(.ckpt 포함): 그대로 사용
    if [[ "${spec}" == *"/"* || "${spec}" == *.ckpt ]]; then
        echo "${spec}"
        return
    fi

    # 스템만: yaml에서 체크포인트 디렉토리를 추출해 완성
    local yaml_file="config/experiment/brainco/ours/task/${exp}.yaml"
    local raw
    raw=$(grep -E '^\s*checkpoint_encoder:' "${yaml_file}" 2>/dev/null | head -1 \
          | sed 's/^\s*checkpoint_encoder:\s*//' \
          | sed 's/\s*#.*//')
    raw="${raw#\$\{paths.encoder_checkpoint_root\}/}"

    if [[ -z "${raw}" || "${raw}" == "~" || "${raw}" == "null" ]]; then
        echo "checkpoints/${spec}.ckpt"
        return
    fi

    local ckpt_dir
    ckpt_dir=$(dirname "${raw}")
    echo "checkpoints/${ckpt_dir}/${spec}.ckpt"
}

# ── 결과 추출 함수 ────────────────────────────────────────────────────────────
# 로그 파일에서 Last/Best Epoch K-Fold summary의 Mean/Std를 모두 추출한다.
# 출력: last_mean_acc,last_std_acc,last_mean_f1,last_std_f1,
#        best_mean_acc,best_std_acc,best_mean_f1,best_std_f1,status
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
        rf'K-FOLD CROSS-VALIDATION SUMMARY \({re.escape(label)}\).*?'
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

# ── Checkpoint 이름 추출 함수 ────────────────────────────────────────────────
# ckpt_override가 있으면 resolve_ckpt로 완성 후 반환, 없으면 yaml에서 읽어 반환.
get_checkpoint_name() {
    local exp="$1"
    local ckpt_override="${2:-}"

    if [[ -n "${ckpt_override}" ]]; then
        resolve_ckpt "${exp}" "${ckpt_override}"
        return
    fi

    local yaml_file="config/experiment/brainco/ours/task/${exp}.yaml"
    if [[ ! -f "${yaml_file}" ]]; then
        echo "yaml_not_found"
        return
    fi

    local raw
    raw=$(grep -E '^\s*checkpoint_encoder:' "${yaml_file}" | head -1 \
          | sed 's/^\s*checkpoint_encoder:\s*//' \
          | sed 's/\s*#.*//')

    if [[ -z "${raw}" || "${raw}" == "~" || "${raw}" == "null" ]]; then
        echo "scratch"
        return
    fi

    raw="${raw#\$\{paths.encoder_checkpoint_root\}/}"
    echo "${raw}"
}

# ── 실행 함수 ─────────────────────────────────────────────────────────────────
run_experiment() {
    local entry="$1"
    local idx="$2"

    parse_entry "${entry}"
    local exp="${PARSED_EXP}"
    local ckpt_override="${PARSED_CKPT}"

    local task_dir="${exp%%/*}"
    local exp_name="${exp##*/}"
    local exp_path="brainco/ours/task/${exp}"
    local log_file="${LOG_DIR}/${task_dir}__${exp_name}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    printf "  [%d/%d]  %s  (started %s)\n" "${idx}" "${TOTAL}" "${exp}" "$(date +%H:%M:%S)"
    local resolved_ckpt
    resolved_ckpt=$(resolve_ckpt "${exp}" "${ckpt_override}")
    [[ -n "${resolved_ckpt}" ]] && printf "  ckpt override: %s\n" "${resolved_ckpt}"
    echo "════════════════════════════════════════════════════════════════"

    # checkpoint override가 있으면 스템 자동완성 후 Hydra override로 전달
    local ckpt_arg=""
    [[ -n "${resolved_ckpt}" ]] && ckpt_arg="task.checkpoint_encoder=${resolved_ckpt}"

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 \
        python train_task_brainco.py \
        "+experiment=${exp_path}" \
        ${ckpt_arg:+"${ckpt_arg}"} \
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

# ── 실험 순차 실행 ────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  실험 시작: ${TOTAL}개  (${TIMESTAMP})"
echo "  로그 디렉토리: ${LOG_DIR}"
echo "════════════════════════════════════════════════════════════════"

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

# CSV 헤더
echo "task,experiment,checkpoint,last_mean_acc,last_std_acc,last_mean_f1,last_std_f1,best_mean_acc,best_std_acc,best_mean_f1,best_std_f1,status" \
    > "${SUMMARY_CSV}"

# 요약 테이블 헤더
HEADER=$(printf "%-25s  %-26s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "Task" "Experiment" "Epoch" "MeanAcc" "StdAcc" "MeanF1" "StdF1" "Status")
SEP=$(printf "%-25s  %-26s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "-------------------------" "--------------------------" "-----" "----------" "----------" "----------" "----------" "------")

{
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════════════"
    printf "  EXPERIMENT RESULTS  (%s)\n" "${TIMESTAMP}"
    echo "════════════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "${HEADER}"
    echo "${SEP}"
} | tee "${SUMMARY_TXT}"

for entry in "${ALL_EXPERIMENTS[@]}"; do
    parse_entry "${entry}"
    exp="${PARSED_EXP}"
    ckpt_override="${PARSED_CKPT}"

    task_dir="${exp%%/*}"
    exp_name="${exp##*/}"
    log_file="${LOG_DIR}/${task_dir}__${exp_name}.log"

    ckpt=$(get_checkpoint_name "${exp}" "${ckpt_override}")
    result=$(extract_results "${log_file}")
    IFS=',' read -r lm_acc ls_acc lm_f1 ls_f1 bm_acc bs_acc bm_f1 bs_f1 status <<< "${result}"

    # txt: checkpoint 표시 행
    printf "%-25s  %-26s  %-5s  %s\n" \
        "${task_dir}" "${exp_name}" "ckpt" "${ckpt}" \
        | tee -a "${SUMMARY_TXT}"

    # txt: Last Epoch 행
    printf "%-25s  %-26s  %-5s  %-10s  %-10s  %-10s  %-10s  %s\n" \
        "" "" "Last" \
        "${lm_acc:-n/a}" "${ls_acc:-n/a}" "${lm_f1:-n/a}" "${ls_f1:-n/a}" \
        "${status:-UNKNOWN}" \
        | tee -a "${SUMMARY_TXT}"

    # txt: Best Epoch 행
    printf "%-25s  %-26s  %-5s  %-10s  %-10s  %-10s  %-10s\n" \
        "" "" "Best" \
        "${bm_acc:-n/a}" "${bs_acc:-n/a}" "${bm_f1:-n/a}" "${bs_f1:-n/a}" \
        | tee -a "${SUMMARY_TXT}"

    echo "${SEP}" | tee -a "${SUMMARY_TXT}"

    # csv: 한 행에 checkpoint + Last + Best 모두
    echo "${task_dir},${exp_name},${ckpt},${lm_acc:-},${ls_acc:-},${lm_f1:-},${ls_f1:-},${bm_acc:-},${bs_acc:-},${bm_f1:-},${bs_f1:-},${status:-UNKNOWN}" \
        >> "${SUMMARY_CSV}"
done

{
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  Logs   : ${LOG_DIR}/"
    echo "  Summary: ${SUMMARY_TXT}"
    echo "  CSV    : ${SUMMARY_CSV}"
    echo "════════════════════════════════════════════════════════════════════════════════"
} | tee -a "${SUMMARY_TXT}"
