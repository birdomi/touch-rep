#!/usr/bin/env bash
# run_angle_experiments.sh
# AngleTransformer + BraincoAngleGraspDataset 기반 grasp prediction 실험을 순차 실행한다.
#
# 사용법:
#   bash scripts/run_angle_experiments.sh                              # scratch + DEFAULT_PRETRAINED_MODELS 전체
#   bash scripts/run_angle_experiments.sh scratch                      # scratch 실험만
#   bash scripts/run_angle_experiments.sh pretrained                   # pretrained 실험만 (DEFAULT_PRETRAINED_MODELS)
#   bash scripts/run_angle_experiments.sh -m epoch-0150                # 특정 모델 하나
#   bash scripts/run_angle_experiments.sh pretrained -m epoch-0150     # pretrained + 특정 모델
#   bash scripts/run_angle_experiments.sh -m "epoch-0150,epoch-0040"   # 여러 모델 (콤마 구분)
#   bash scripts/run_angle_experiments.sh -m "dinov2_angle/epoch-0150.ckpt,dinov2_multi_sensor_pretrained/epoch-0150-all-cls.ckpt"
#   bash scripts/run_angle_experiments.sh pretrained -m yaml           # yaml 기본 checkpoint만 사용
#
# 결과:
#   scripts/logs/<timestamp>/  — 실험별 전체 로그
#   scripts/results_angle_<timestamp>.txt  — 요약 테이블
#   scripts/results_angle_<timestamp>.csv  — CSV (spreadsheet용)

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/${TIMESTAMP}"
SUMMARY_TXT="${SCRIPT_DIR}/results_angle_${TIMESTAMP}.txt"
SUMMARY_CSV="${SCRIPT_DIR}/results_angle_${TIMESTAMP}.csv"

mkdir -p "${LOG_DIR}"

# ── 실험 목록 ─────────────────────────────────────────────────────────────────
SCRATCH_EXPERIMENTS=(
    # "grasp_prediction/dinov2_multi_scratch"
    "grasp_prediction/dinov2_multi_scratch_combined"
)

PRETRAINED_EXPERIMENTS=(
    # "grasp_prediction/dinov2_multi"
    # "grasp_prediction/dinov2_multi_freeze_combined"
    # "grasp_prediction/dinov2_multi_mask"
    # "grasp_prediction/dinov2_multi_mask_freeze"
    # "grasp_prediction/dinov2_multi_combined"
)

# -m/--models를 생략하면 아래 체크포인트들을 pretrained 실험에 사용한다.
# - "epoch-XXXX"처럼 stem만 쓰면 각 experiment yaml의 checkpoint_encoder 디렉토리를 따라간다.
# - "dir/file.ckpt"처럼 경로를 쓰면 그대로 사용한다.
# - yaml 기본값만 쓰고 싶으면 CLI에서 `-m yaml`을 넘긴다.
DEFAULT_PRETRAINED_MODELS=(
    # "epoch-0200-all"
    # "epoch-0110-all"
    # "epoch-0150"
    # "epoch-0200-hot3d"
    # "epoch-0200-taco"
    # "epoch-0300-arctic"
    # "epoch-5000-brainco"
)

# ── CLI 인자 파싱 ─────────────────────────────────────────────────────────────
# 사용 가능한 옵션:
#   [scratch|pretrained|all]   실험 범위 (기본: all)
#   -m|--models <m1,m2,...>    pretrained 실험에 사용할 체크포인트 목록
MODE="all"
SELECTED_MODELS=()
USING_DEFAULT_MODELS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        scratch|pretrained|all)
            MODE="$1"
            shift
            ;;
        -m|--models)
            shift
            IFS=',' read -ra SELECTED_MODELS <<< "$1"
            USING_DEFAULT_MODELS=false
            shift
            ;;
        *)
            echo "[ERROR] 알 수 없는 인자: $1" >&2
            echo "  사용법: $0 [scratch|pretrained|all] [-m model1,model2,...]" >&2
            exit 1
            ;;
    esac
done

# 모델 미지정 시 스크립트 내부 DEFAULT_PRETRAINED_MODELS 사용
if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    if [[ ${#DEFAULT_PRETRAINED_MODELS[@]} -gt 0 ]]; then
        SELECTED_MODELS=("${DEFAULT_PRETRAINED_MODELS[@]}")
    else
        SELECTED_MODELS=("yaml")
    fi
fi

# "yaml" = yaml 기본 checkpoint 사용 (빈 문자열 = 오버라이드 없음)
for i in "${!SELECTED_MODELS[@]}"; do
    if [[ "${SELECTED_MODELS[$i]}" == "yaml" || "${SELECTED_MODELS[$i]}" == "default" ]]; then
        SELECTED_MODELS[$i]=""
    fi
done

# ── ALL_EXPERIMENTS 동적 생성 ────────────────────────────────────────────────
# scratch: 모델 선택과 무관하게 1회 실행
# pretrained: SELECTED_MODELS × PRETRAINED_EXPERIMENTS 조합
ALL_EXPERIMENTS=()

if [[ "${MODE}" == "all" || "${MODE}" == "scratch" ]]; then
    for exp in "${SCRATCH_EXPERIMENTS[@]}"; do
        ALL_EXPERIMENTS+=("${exp}:")   # 모델 없음
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

# ── 엔트리 파싱 헬퍼 ──────────────────────────────────────────────────────────
# 형식: "task/exp_name:model_spec"  (model_spec은 빈 문자열 가능)
parse_entry() {
    local entry="$1"
    PARSED_EXP="${entry%%:*}"
    PARSED_CKPT="${entry#*:}"   # 콜론 이후 전부 (없으면 빈 문자열)
}

# ── 체크포인트 경로 해석 ──────────────────────────────────────────────────────
# spec이 비어 있으면 yaml의 checkpoint_encoder 그대로 사용 → 빈 문자열 반환
# spec이 "epoch-XXXX" 형식이면 yaml에서 디렉토리를 추출해 조합
# spec이 경로(/ 포함) 또는 .ckpt 이면 그대로 사용
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

# ── 체크포인트 표시 이름 ──────────────────────────────────────────────────────
# 결과 출력 시 사람이 읽기 좋은 이름으로 변환
display_model() {
    local exp="$1"
    local spec="$2"

    if [[ -z "${spec}" ]]; then
        # yaml 기본값 읽기
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

    # 오버라이드된 경우 resolve해서 표시
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
    def extract_single(label):
        pat = rf'{label}\s+[^A-Za-z0-9]*\s+Acc:\s+([\d.]+|nan)\s+F1:\s+([\d.]+|nan)'
        m = re.search(pat, text)
        if m:
            # Single explicit-val runs do not have fold std.
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

    # 모델별로 로그 파일을 분리 (같은 실험을 여러 모델로 돌릴 때 덮어쓰기 방지)
    local model_tag=""
    if [[ -n "${ckpt_spec}" ]]; then
        model_tag="__$(basename "${ckpt_spec}" .ckpt | tr '/' '_')"
    fi
    local log_file="${LOG_DIR}/${task_dir}__${exp_name}${model_tag}.log"

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

# ── 선택된 모델 목록 출력 ─────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  [Angle Grasp] 실험 시작: ${TOTAL}개  (${TIMESTAMP})"
echo "  로그 디렉토리: ${LOG_DIR}"
echo "  모드: ${MODE}"
if [[ "${SELECTED_MODELS[0]}" == "" && ${#SELECTED_MODELS[@]} -eq 1 ]]; then
    echo "  pretrained model: (각 yaml 기본값 사용)"
else
    if [[ "${USING_DEFAULT_MODELS}" == true ]]; then
        echo "  pretrained model(s): (DEFAULT_PRETRAINED_MODELS)"
    else
        echo "  pretrained model(s):"
    fi
    for m in "${SELECTED_MODELS[@]}"; do
        echo "    - ${m}"
    done
fi
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

echo "task,experiment,pretrained_model,last_mean_acc,last_mean_f1,last_std_acc,last_std_f1,best_mean_acc,best_mean_f1,best_std_acc,best_std_f1,status" \
    > "${SUMMARY_CSV}"

HEADER=$(printf "%-25s  %-35s  %-5s  %-10s  %-10s  %-10s  %-10s  %s" \
    "Task" "Experiment" "Epoch" "MeanAcc" "MeanF1" "StdAcc" "StdF1" "Status")
SEP=$(printf "%s" \
    "-------------------------  -----------------------------------  -----  ----------  ----------  ----------  ----------  ------")

{
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════════════════════"
    printf "  ANGLE GRASP PREDICTION RESULTS  (%s)\n" "${TIMESTAMP}"
    echo "══════════════════════════════════════════════════════════════════════════════════════════"

    # ── 사용된 pretrained model 목록 ────────────────────────────────────────
    echo ""
    if [[ "${SELECTED_MODELS[0]}" == "" && ${#SELECTED_MODELS[@]} -eq 1 ]]; then
        echo "  Pretrained model: (각 yaml 기본값)"
    else
        if [[ "${USING_DEFAULT_MODELS}" == true ]]; then
            echo "  Pretrained model(s) used: (DEFAULT_PRETRAINED_MODELS)"
        else
            echo "  Pretrained model(s) used:"
        fi
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

    task_dir="${exp%%/*}"
    exp_name="${exp##*/}"

    model_tag=""
    if [[ -n "${ckpt_spec}" ]]; then
        model_tag="__$(basename "${ckpt_spec}" .ckpt | tr '/' '_')"
    fi
    log_file="${LOG_DIR}/${task_dir}__${exp_name}${model_tag}.log"

    model_display=$(display_model "${exp}" "${ckpt_spec}")
    result=$(extract_results "${log_file}")
    IFS=',' read -r lm_acc lm_f1 ls_acc ls_f1 bm_acc bm_f1 bs_acc bs_f1 status <<< "${result}"

    # 모델이 바뀔 때 구분선 + 모델 헤더 출력
    if [[ "${model_display}" != "${prev_model}" ]]; then
        {
            echo ""
            printf "  [Pretrained: %s]\n" "${model_display}"
        } | tee -a "${SUMMARY_TXT}"
        prev_model="${model_display}"
    fi

    printf "%-25s  %-35s  %-5s  %-10s  %-10s  %-10s  %-10s  %s\n" \
        "${task_dir}" "${exp_name}" "Last" \
        "${lm_acc:-n/a}" "${lm_f1:-n/a}" "${ls_acc:-n/a}" "${ls_f1:-n/a}" \
        "${status:-UNKNOWN}" \
        | tee -a "${SUMMARY_TXT}"

    printf "%-25s  %-35s  %-5s  %-10s  %-10s  %-10s  %-10s\n" \
        "" "" "Best" \
        "${bm_acc:-n/a}" "${bm_f1:-n/a}" "${bs_acc:-n/a}" "${bs_f1:-n/a}" \
        | tee -a "${SUMMARY_TXT}"

    echo "${task_dir},${exp_name},${model_display},${lm_acc:-},${lm_f1:-},${ls_acc:-},${ls_f1:-},${bm_acc:-},${bm_f1:-},${bs_acc:-},${bs_f1:-},${status:-UNKNOWN}" \
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
