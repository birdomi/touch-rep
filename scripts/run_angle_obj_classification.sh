#!/usr/bin/env bash
# run_angle_obj_classification.sh
# AngleTransformer 기반 object classification 실험을 순차 실행한다.
#
# 사용법:
#   bash scripts/run_angle_obj_classification.sh                    # scratch + pretrained(hot3d/taco/arctic/brainco) 전체
#   bash scripts/run_angle_obj_classification.sh scratch            # scratch 실험만
#   bash scripts/run_angle_obj_classification.sh pretrained         # pretrained × {hot3d,taco,arctic,brainco}
#   bash scripts/run_angle_obj_classification.sh -m epoch-0150      # 특정 모델 하나
#   bash scripts/run_angle_obj_classification.sh pretrained -m "epoch-0150,epoch-0040"
#   bash scripts/run_angle_obj_classification.sh -m "dinov2_angle/epoch-0150.ckpt,dinov2_multi_sensor_pretrained/epoch-0150-all-cls.ckpt"
#
# 기본 pretrained 모델 (DEFAULT_PRETRAINED_MODELS):
#   epoch-0200-hot3d, epoch-0200-taco, epoch-0300-arctic, epoch-5000-brainco
#   (모두 checkpoints/dinov2_angle/ 하위)
#
# 결과:
#   scripts/logs/<timestamp>/  — 실험별 전체 로그
#   scripts/results_angle_oc_<timestamp>.txt  — 요약 테이블
#   scripts/results_angle_oc_<timestamp>.csv  — CSV (spreadsheet용)

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
SUMMARY_TXT="${SCRIPT_DIR}/results_angle_oc_${TIMESTAMP}.txt"
SUMMARY_CSV="${SCRIPT_DIR}/results_angle_oc_${TIMESTAMP}.csv"

mkdir -p "${LOG_DIR}"

# ── 실험 목록 ─────────────────────────────────────────────────────────────────
SCRATCH_EXPERIMENTS=(
    "object_classification/dinov2_angle_scratch"
)

PRETRAINED_EXPERIMENTS=(
    "object_classification/dinov2_angle"
    "object_classification/dinov2_angle_freeze"
    "object_classification/dinov2_multi"
    "object_classification/dinov2_multi_freeze"
)

# 기본 pretrained 모델 목록: -m 미지정 시 dinov2_angle/ 의 네 체크포인트를 모두 순회한다.
DEFAULT_PRETRAINED_MODELS=(
    "epoch-0200-hot3d"
    "epoch-0200-taco"
    "epoch-0300-arctic"
    "epoch-5000-brainco"
)

# ── CLI 인자 파싱 ─────────────────────────────────────────────────────────────
MODE="all"
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        scratch|pretrained|all)
            MODE="$1"
            shift
            ;;
        -m|--models)
            shift
            IFS=',' read -ra SELECTED_MODELS <<< "$1"
            shift
            ;;
        *)
            echo "[ERROR] 알 수 없는 인자: $1" >&2
            echo "  사용법: $0 [scratch|pretrained|all] [-m model1,model2,...]" >&2
            exit 1
            ;;
    esac
done

# 모델 미지정 시 DEFAULT_PRETRAINED_MODELS 의 4개 체크포인트(hot3d/taco/arctic/brainco)를 순회한다.
[[ ${#SELECTED_MODELS[@]} -eq 0 ]] && SELECTED_MODELS=("${DEFAULT_PRETRAINED_MODELS[@]}")

# ── ALL_EXPERIMENTS 동적 생성 ─────────────────────────────────────────────────
ALL_EXPERIMENTS=()

if [[ "${MODE}" == "all" || "${MODE}" == "scratch" ]]; then
    for exp in "${SCRATCH_EXPERIMENTS[@]}"; do
        ALL_EXPERIMENTS+=("${exp}:")
    done
fi

if [[ "${MODE}" == "all" || "${MODE}" == "pretrained" ]]; then
    for model in "${SELECTED_MODELS[@]}"; do
        for exp in "${PRETRAINED_EXPERIMENTS[@]}"; do
            ALL_EXPERIMENTS+=("${exp}:${model}")
        done
    done
fi

TOTAL=${#ALL_EXPERIMENTS[@]}

# ── 헬퍼 함수들 ───────────────────────────────────────────────────────────────
parse_entry() {
    local entry="$1"
    PARSED_EXP="${entry%%:*}"
    PARSED_CKPT="${entry#*:}"
}

resolve_ckpt() {
    local exp="$1"
    local spec="$2"

    [[ -z "${spec}" ]] && echo "" && return

    if [[ "${spec}" == *"/"* || "${spec}" == *.ckpt ]]; then
        echo "${spec}"
        return
    fi

    local yaml_file="config/experiment/brainco/ours_vectors/task/${exp}.yaml"
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

display_model() {
    local exp="$1"
    local spec="$2"

    if [[ -z "${spec}" ]]; then
        local yaml_file="config/experiment/brainco/ours_vectors/task/${exp}.yaml"
        local raw
        raw=$(grep -E '^\s*checkpoint_encoder:' "${yaml_file}" 2>/dev/null | head -1 \
              | sed 's/^\s*checkpoint_encoder:\s*//' \
              | sed 's/\s*#.*//')
        raw="${raw#\$\{paths.encoder_checkpoint_root\}/}"
        if [[ -z "${raw}" || "${raw}" == "~" || "${raw}" == "null" ]]; then
            echo "scratch"
        else
            echo "${raw}"
        fi
        return
    fi

    local resolved
    resolved=$(resolve_ckpt "${exp}" "${spec}")
    echo "${resolved:-${spec}}"
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

    local exp_name="${exp##*/}"
    local exp_path="brainco/ours_vectors/task/${exp}"

    local model_tag=""
    if [[ -n "${ckpt_spec}" ]]; then
        model_tag="__$(basename "${ckpt_spec}" .ckpt | tr '/' '_')"
    fi
    local log_file="${LOG_DIR}/object_classification__${exp_name}${model_tag}.log"

    local resolved_ckpt
    resolved_ckpt=$(resolve_ckpt "${exp}" "${ckpt_spec}")

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    printf "  [%d/%d]  %s  (started %s)\n" "${idx}" "${TOTAL}" "${exp}" "$(date +%H:%M:%S)"
    if [[ -n "${resolved_ckpt}" ]]; then
        printf "  pretrained model: %s\n" "${resolved_ckpt}"
    else
        printf "  pretrained model: (yaml 기본값)\n"
    fi
    echo "════════════════════════════════════════════════════════════════"

    local ckpt_arg=""
    [[ -n "${resolved_ckpt}" ]] && ckpt_arg="task.checkpoint_encoder=${resolved_ckpt}"

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 \
        python train_task_brainco_angle.py \
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

# ── 실행 ──────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  [Angle OC] 실험 시작: ${TOTAL}개  (${TIMESTAMP})"
echo "  로그 디렉토리: ${LOG_DIR}"
echo "  모드: ${MODE}"
if [[ "${SELECTED_MODELS[0]}" == "" && ${#SELECTED_MODELS[@]} -eq 1 ]]; then
    echo "  pretrained model: (각 yaml 기본값 사용)"
else
    echo "  pretrained model(s):"
    for m in "${SELECTED_MODELS[@]}"; do
        echo "    - ${m}"
    done
fi
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

echo "experiment,pretrained_model,last_mean_acc,last_std_acc,last_mean_f1,last_std_f1,best_mean_acc,best_std_acc,best_mean_f1,best_std_f1,status" \
    > "${SUMMARY_CSV}"

HEADER=$(printf "%-38s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "Experiment" "Epoch" "MeanAcc" "StdAcc" "MeanF1" "StdF1" "Status")
SEP=$(printf "%s" \
    "--------------------------------------  -----  ----------  ----------  ----------  ----------  ------")

{
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════════════════════"
    printf "  ANGLE OBJECT CLASSIFICATION RESULTS  (%s)\n" "${TIMESTAMP}"
    echo "══════════════════════════════════════════════════════════════════════════════════════════"
    echo ""
    if [[ "${SELECTED_MODELS[0]}" == "" && ${#SELECTED_MODELS[@]} -eq 1 ]]; then
        echo "  Pretrained model: (각 yaml 기본값)"
    else
        echo "  Pretrained model(s) used:"
        for m in "${SELECTED_MODELS[@]}"; do
            echo "    - ${m}"
        done
    fi
    echo ""
    echo "${HEADER}"
    echo "${SEP}"
} | tee "${SUMMARY_TXT}"

prev_model=""
for entry in "${ALL_EXPERIMENTS[@]}"; do
    parse_entry "${entry}"
    exp="${PARSED_EXP}"
    ckpt_spec="${PARSED_CKPT}"

    exp_name="${exp##*/}"
    model_tag=""
    if [[ -n "${ckpt_spec}" ]]; then
        model_tag="__$(basename "${ckpt_spec}" .ckpt | tr '/' '_')"
    fi
    log_file="${LOG_DIR}/object_classification__${exp_name}${model_tag}.log"

    model_display=$(display_model "${exp}" "${ckpt_spec}")
    result=$(extract_results "${log_file}")
    IFS=',' read -r lm_acc ls_acc lm_f1 ls_f1 bm_acc bs_acc bm_f1 bs_f1 status <<< "${result}"

    if [[ "${model_display}" != "${prev_model}" ]]; then
        {
            echo ""
            printf "  [Pretrained: %s]\n" "${model_display}"
        } | tee -a "${SUMMARY_TXT}"
        prev_model="${model_display}"
    fi

    printf "%-38s  %-5s  %-10s  %-10s  %-10s  %-10s  %s\n" \
        "${exp_name}" "Last" \
        "${lm_acc:-n/a}" "${ls_acc:-n/a}" "${lm_f1:-n/a}" "${ls_f1:-n/a}" \
        "${status:-UNKNOWN}" \
        | tee -a "${SUMMARY_TXT}"

    printf "%-38s  %-5s  %-10s  %-10s  %-10s  %-10s\n" \
        "" "Best" \
        "${bm_acc:-n/a}" "${bs_acc:-n/a}" "${bm_f1:-n/a}" "${bs_f1:-n/a}" \
        | tee -a "${SUMMARY_TXT}"

    echo "${exp_name},${model_display},${lm_acc:-},${ls_acc:-},${lm_f1:-},${ls_f1:-},${bm_acc:-},${bs_acc:-},${bm_f1:-},${bs_f1:-},${status:-UNKNOWN}" \
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
