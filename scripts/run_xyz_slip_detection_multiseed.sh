#!/usr/bin/env bash
# Run pretrained and scratch BrainCo XYZ slip-detection fine-tuning across seeds.
#
# Usage:
#   bash scripts/run_xyz_slip_detection_multiseed.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_slip_detection_multiseed.sh
#   SEEDS="0 1 2" RUN_SCRATCH=0 bash scripts/run_xyz_slip_detection_multiseed.sh
#   SEEDS="0 1 2" RUN_PRETRAINED=0 bash scripts/run_xyz_slip_detection_multiseed.sh
#   SEEDS="7" NUM_FOLDS=5 bash scripts/run_xyz_slip_detection_multiseed.sh
#   SEEDS="0 1 2" RUN_ALL_SPLITS=0 bash scripts/run_xyz_slip_detection_multiseed.sh
#   SEEDS="0 1 2" bash scripts/run_xyz_slip_detection_multiseed.sh -- trainer.max_epochs=50
#
# The model/training RNG changes per item in SEEDS.  The episode split remains
# fixed (SPLIT_SEED=42 by default), making seed-to-seed scores comparable.

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT:-brainco/ours_3d/task/slip_detection/dinov2_all_rope}"
SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT:-brainco/ours_3d/task/slip_detection/dinov2_all_rope_scratch}"
SEEDS_STRING="${SEEDS:-0 1 2}"
read -r -a SEED_LIST <<< "${SEEDS_STRING}"
SPLIT_SEED="${SPLIT_SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-4}"
RUN_ALL_SPLITS="${RUN_ALL_SPLITS:-1}"
RUN_PRETRAINED="${RUN_PRETRAINED:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
EXTRA_OVERRIDES=()

if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        echo "[ERROR] Extra Hydra overrides must follow --" >&2
        echo "Usage: $0 -- key=value key2=value2" >&2
        exit 1
    fi
    shift
    EXTRA_OVERRIDES=("$@")
fi

if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] SEEDS must contain at least one integer." >&2
    exit 1
fi
if [[ "${RUN_ALL_SPLITS}" != "0" && "${RUN_ALL_SPLITS}" != "1" ]]; then
    echo "[ERROR] RUN_ALL_SPLITS must be 0 or 1." >&2
    exit 1
fi
if [[ "${RUN_PRETRAINED}" != "1" && "${RUN_SCRATCH}" != "1" ]]; then
    echo "[ERROR] Enable at least one of RUN_PRETRAINED=1 or RUN_SCRATCH=1." >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/xyz_slip_detection_multiseed_${TIMESTAMP}"
RESULT_CSV="${SCRIPT_DIR}/results_xyz_slip_detection_multiseed_${TIMESTAMP}.csv"
RESULT_MD="${SCRIPT_DIR}/results_xyz_slip_detection_multiseed_${TIMESTAMP}.md"
mkdir -p "${LOG_DIR}"

echo "model,seed,last_acc,last_f1,best_acc,best_f1,status,log_file" > "${RESULT_CSV}"

extract_result() {
    local log_file="$1"
    local aggregate="$2"
    python3 - "${log_file}" "${aggregate}" <<'PYEOF'
import re
import sys

log_path, aggregate = sys.argv[1:]
try:
    text = open(log_path).read()
except OSError:
    print(",,,,NOFILE")
    raise SystemExit

if "SLIP_EXPERIMENT_FAILED" in text:
    print(",,,,FAILED")
    raise SystemExit

def metric_block(label):
    if aggregate == "1":
        match = re.search(
            rf"K-FOLD SUMMARY \({label} Epoch\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )
    else:
        match = re.search(
            rf"{label}\s+[^A-Za-z0-9]*\s+Acc:\s+([\d.]+|nan)\s+F1:\s+([\d.]+|nan)",
            text,
        )
    return match.groups() if match else ("", "")

last_acc, last_f1 = metric_block("Last")
best_acc, best_f1 = metric_block("Best")
status = "OK" if any((last_acc, last_f1, best_acc, best_f1)) else "INCOMPLETE"
print(",".join((last_acc, last_f1, best_acc, best_f1, status)))
PYEOF
}

echo "XYZ DINOv2 slip-detection multi-seed experiments"
echo "  pretrained: ${PRETRAINED_EXPERIMENT} (enabled: ${RUN_PRETRAINED})"
echo "  scratch:    ${SCRATCH_EXPERIMENT} (enabled: ${RUN_SCRATCH})"
echo "  seeds:      ${SEEDS_STRING}"
echo "  split seed: ${SPLIT_SEED}"
echo "  folds:      ${NUM_FOLDS} (all folds: ${RUN_ALL_SPLITS})"
echo "  logs:       ${LOG_DIR}"
echo ""

run_experiment() {
    local model_name="$1"
    local experiment="$2"
    local seed="$3"
    local exp_name="${experiment##*/}"
    local log_file="${LOG_DIR}/${model_name}_seed_${seed}.log"
    local run_name="xyz_slip_detection_${exp_name}_${model_name}_seed_${seed}"
    cmd=(
        python train_task_brainco_angle.py
        "+experiment=${experiment}"
        "seed=${seed}"
        "split_seed=${SPLIT_SEED}"
        "experiment_name=${run_name}"
        "wandb.group=xyz_slip_detection_${exp_name}_${model_name}_multiseed"
        "wandb.tags=[brainco,xyz,dinov2,slip_detection,${model_name},multiseed,seed_${seed}]"
        "${EXTRA_OVERRIDES[@]}"
    )
    if [[ "${RUN_ALL_SPLITS}" == "1" ]]; then
        cmd+=(--all_split --num_folds "${NUM_FOLDS}")
    fi

    echo "================================================================"
    echo "  model: ${model_name}  seed: ${seed}"
    echo "  log:  ${log_file}"
    echo "================================================================"
    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 "${cmd[@]}" 2>&1 | tee "${log_file}"; then
        status="OK"
    else
        status="FAILED"
        echo "SLIP_EXPERIMENT_FAILED" >> "${log_file}"
    fi

    result="$(extract_result "${log_file}" "${RUN_ALL_SPLITS}")"
    IFS=',' read -r last_acc last_f1 best_acc best_f1 parsed_status <<< "${result}"
    [[ "${status}" == "OK" ]] || parsed_status="FAILED"
    echo "${model_name},${seed},${last_acc},${last_f1},${best_acc},${best_f1},${parsed_status},${log_file}" >> "${RESULT_CSV}"
}

for seed in "${SEED_LIST[@]}"; do
    if [[ "${RUN_PRETRAINED}" == "1" ]]; then
        run_experiment "pretrained" "${PRETRAINED_EXPERIMENT}" "${seed}"
    fi
    if [[ "${RUN_SCRATCH}" == "1" ]]; then
        run_experiment "scratch" "${SCRATCH_EXPERIMENT}" "${seed}"
    fi
done

python3 - "${RESULT_CSV}" "${RESULT_MD}" <<'PYEOF'
import csv
import math
import statistics
import sys

csv_path, markdown_path = sys.argv[1:]
rows = list(csv.DictReader(open(csv_path)))
ok_rows = [row for row in rows if row["status"] == "OK"]

def values(key, model):
    parsed = []
    for row in ok_rows:
        if row["model"] != model:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            parsed.append(value)
    return parsed

def mean_std(key, model):
    data = values(key, model)
    if not data:
        return "n/a"
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    return f"{statistics.mean(data):.4f} ± {std:.4f}"

lines = [
    "# XYZ DINOv2 Slip-detection Multi-seed Results",
    "",
    "| Model | Seed | Last Acc | Last F1 | Best Acc | Best F1 | Status | Log |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
for row in rows:
    lines.append(
        f"| {row['model']} | {row['seed']} | {row['last_acc'] or 'n/a'} | {row['last_f1'] or 'n/a'} | "
        f"{row['best_acc'] or 'n/a'} | {row['best_f1'] or 'n/a'} | {row['status']} | "
        f"`{row['log_file']}` |"
    )
lines.extend([
    "",
    "## Mean ± sample standard deviation",
    "",
    "| Model | Last Acc | Last F1 | Best Acc | Best F1 |",
    "| --- | ---: | ---: | ---: | ---: |",
])
for model in sorted({row["model"] for row in rows}):
    lines.append(
        f"| {model} | {mean_std('last_acc', model)} | {mean_std('last_f1', model)} | "
        f"{mean_std('best_acc', model)} | {mean_std('best_f1', model)} |"
    )
open(markdown_path, "w").write("\n".join(lines) + "\n")
PYEOF

echo ""
echo "Done."
echo "  Per-seed metrics: ${RESULT_CSV}"
echo "  Aggregated report: ${RESULT_MD}"
