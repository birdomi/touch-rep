#!/usr/bin/env bash
# run_all_experiments_object_classification.sh
# BrainCo object classification 실험을 순차 실행하고 K-Fold 결과를 취합해 저장한다.
#
# 사용법:
#   bash scripts/run_all_experiments_object_classification.sh
#
# 결과:
#   scripts/logs/<timestamp>/  — 실험별 전체 로그
#   scripts/results_oc_<timestamp>.txt  — 요약 테이블
#   scripts/results_oc_<timestamp>.csv  — CSV (spreadsheet용)

# sh로 실행 시 bash로 재실행
[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
SUMMARY_TXT="${SCRIPT_DIR}/results_oc_${TIMESTAMP}.txt"
SUMMARY_CSV="${SCRIPT_DIR}/results_oc_${TIMESTAMP}.csv"

mkdir -p "${LOG_DIR}"

# ── 실험 목록 ─────────────────────────────────────────────────────────────────
# 형식 1: "object_classification/experiment_yaml_stem"
#          → yaml 파일의 checkpoint_encoder 그대로 사용
# 형식 2: "object_classification/experiment_yaml_stem:ckpt_path"
#          → ckpt_path로 직접 지정 (프로젝트 루트 기준 상대경로 또는 절대경로)
# 주석 처리하면 해당 실험 스킵

EXPERIMENTS=(
    "object_classification/dinov2_multi_scratch"
    "object_classification/dinov2_from_scratch"
    "object_classification/dinov2_multi"
    "object_classification/dinov2_multi_mask"
    "object_classification/dinov2_multi_mask_rm"
    # "object_classification/dinov2_multi:checkpoints/dinov2_multi_sensor_pretrained/epoch-0300-taco.ckpt"
)

ALL_EXPERIMENTS=("${EXPERIMENTS[@]}")
TOTAL=${#ALL_EXPERIMENTS[@]}

# ── 엔트리 파싱 헬퍼 ─────────────────────────────────────────────────────────
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
get_checkpoint_name() {
    local exp="$1"
    local ckpt_override="${2:-}"

    if [[ -n "${ckpt_override}" ]]; then
        echo "${ckpt_override}"
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

    local exp_name="${exp##*/}"
    local exp_path="brainco/ours/task/${exp}"
    local log_file="${LOG_DIR}/object_classification__${exp_name}.log"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    printf "  [%d/%d]  %s  (started %s)\n" "${idx}" "${TOTAL}" "${exp}" "$(date +%H:%M:%S)"
    [[ -n "${ckpt_override}" ]] && printf "  ckpt override: %s\n" "${ckpt_override}"
    echo "════════════════════════════════════════════════════════════════"

    local ckpt_arg=""
    [[ -n "${ckpt_override}" ]] && ckpt_arg="task.checkpoint_encoder=${ckpt_override}"

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 \
        python train_task_brainco_all.py \
        "+experiment=${exp_path}" \
        ${ckpt_arg:+"${ckpt_arg}"} \
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
echo "  Object Classification 실험 시작: ${TOTAL}개  (${TIMESTAMP})"
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

echo "experiment,checkpoint,last_mean_acc,last_std_acc,last_mean_f1,last_std_f1,best_mean_acc,best_std_acc,best_mean_f1,best_std_f1,status" \
    > "${SUMMARY_CSV}"

HEADER=$(printf "%-30s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "Experiment" "Epoch" "MeanAcc" "StdAcc" "MeanF1" "StdF1" "Status")
SEP=$(printf "%-30s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "------------------------------" "-----" "----------" "----------" "----------" "----------" "------")

{
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════════════"
    printf "  OBJECT CLASSIFICATION RESULTS  (%s)\n" "${TIMESTAMP}"
    echo "════════════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "${HEADER}"
    echo "${SEP}"
} | tee "${SUMMARY_TXT}"

for entry in "${ALL_EXPERIMENTS[@]}"; do
    parse_entry "${entry}"
    exp="${PARSED_EXP}"
    ckpt_override="${PARSED_CKPT}"

    exp_name="${exp##*/}"
    log_file="${LOG_DIR}/object_classification__${exp_name}.log"

    ckpt=$(get_checkpoint_name "${exp}" "${ckpt_override}")
    result=$(extract_results "${log_file}")
    IFS=',' read -r lm_acc ls_acc lm_f1 ls_f1 bm_acc bs_acc bm_f1 bs_f1 status <<< "${result}"

    printf "%-30s  %-5s  %s\n" "${exp_name}" "ckpt" "${ckpt}" \
        | tee -a "${SUMMARY_TXT}"

    printf "%-30s  %-5s  %-10s  %-10s  %-10s  %-10s  %s\n" \
        "" "Last" \
        "${lm_acc:-n/a}" "${ls_acc:-n/a}" "${lm_f1:-n/a}" "${ls_f1:-n/a}" \
        "${status:-UNKNOWN}" \
        | tee -a "${SUMMARY_TXT}"

    printf "%-30s  %-5s  %-10s  %-10s  %-10s  %-10s\n" \
        "" "Best" \
        "${bm_acc:-n/a}" "${bs_acc:-n/a}" "${bm_f1:-n/a}" "${bs_f1:-n/a}" \
        | tee -a "${SUMMARY_TXT}"

    echo "${SEP}" | tee -a "${SUMMARY_TXT}"

    echo "${exp_name},${ckpt},${lm_acc:-},${ls_acc:-},${lm_f1:-},${ls_f1:-},${bm_acc:-},${bs_acc:-},${bm_f1:-},${bs_f1:-},${status:-UNKNOWN}" \
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
