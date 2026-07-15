#!/usr/bin/env bash
# Object-wise 3:1 train/val runner for BrainCo angle grasp prediction with RoPE.
#
# Runs four leave-one-object-out experiments:
#   train = three objects, val = one held-out object
#
# Usage:
#   bash scripts/run_angle_rope_objectwise_grasp.sh
#   bash scripts/run_angle_rope_objectwise_grasp.sh brainco/ours_vectors/task/grasp_prediction_rope/dinov2_combined_scratch
#   bash scripts/run_angle_rope_objectwise_grasp.sh brainco/ours_vectors/task/grasp_prediction_rope/dinov2_combined -- task.checkpoint_encoder=checkpoints/dinov2_angle_rope/checkpoints/epoch-0500-rope.ckpt

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

EXPERIMENT="brainco/ours_vectors/task/grasp_prediction_rope/dinov2_combined"
EXTRA_OVERRIDES=()

if [[ $# -gt 0 && "$1" != "--" ]]; then
    EXPERIMENT="$1"
    shift
fi

if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        echo "[ERROR] Extra Hydra overrides must come after --" >&2
        echo "Usage: $0 [experiment] -- key=value key2=value2" >&2
        exit 1
    fi
    shift
    EXTRA_OVERRIDES=("$@")
fi

OBJECTS=(box tumbler eraser driver)
DATA_ROOT='${paths.data_root}/brainco/downstream/grasp_prediction_0611'
DATASET_TARGET="tactile_ssl.data.brainco_angle_grasp_dataset.BraincoAngleGraspDataset"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="${EXPERIMENT##*/}"
LOG_DIR="${SCRIPT_DIR}/logs/rope_objectwise_${EXP_NAME}_${TIMESTAMP}"
SUMMARY_CSV="${SCRIPT_DIR}/results_angle_rope_objectwise_${EXP_NAME}_${TIMESTAMP}.csv"
SUMMARY_TXT="${SCRIPT_DIR}/results_angle_rope_objectwise_${EXP_NAME}_${TIMESTAMP}.txt"

mkdir -p "${LOG_DIR}"

label_dirs_for_objects() {
    local out=""
    local obj
    for obj in "$@"; do
        [[ -n "${out}" ]] && out+=","
        out+="${obj}_succ:1,${obj}_fail:0"
    done
    echo "${out}"
}

train_objects_for_val() {
    local val_obj="$1"
    local out=()
    local obj
    for obj in "${OBJECTS[@]}"; do
        [[ "${obj}" == "${val_obj}" ]] && continue
        out+=("${obj}")
    done
    echo "${out[@]}"
}

extract_result() {
    local log_file="$1"
    python3 - "${log_file}" <<'PYEOF'
import re
import sys

path = sys.argv[1]
try:
    text = open(path).read()
except OSError:
    print(",,,,NOFILE")
    raise SystemExit

if "OBJECTWISE_EXPERIMENT_FAILED" in text:
    print(",,,,FAILED")
    raise SystemExit

def extract(label):
    pat = rf'{label}\s+[^A-Za-z0-9]*\s+Acc:\s+([\d.]+|nan)\s+F1:\s+([\d.]+|nan)'
    match = re.search(pat, text)
    if match:
        return match.group(1), match.group(2)
    return "", ""

last_acc, last_f1 = extract("Last")
best_acc, best_f1 = extract("Best")
status = "OK" if any([last_acc, last_f1, best_acc, best_f1]) else "INCOMPLETE"
print(",".join([last_acc, last_f1, best_acc, best_f1, status]))
PYEOF
}

echo "Object-wise angle RoPE grasp experiments"
echo "  experiment: ${EXPERIMENT}"
echo "  data root:  ${DATA_ROOT}"
echo "  log dir:    ${LOG_DIR}"
echo "  summary:    ${SUMMARY_TXT}"
echo ""

echo "val_object,train_objects,last_acc,last_f1,best_acc,best_f1,status,log_file" > "${SUMMARY_CSV}"

{
    echo "Object-wise Angle RoPE Grasp Results (${TIMESTAMP})"
    echo "Experiment: ${EXPERIMENT}"
    echo ""
    printf "%-10s  %-26s  %-8s  %-8s  %-8s  %-8s  %s\n" \
        "ValObject" "TrainObjects" "LastAcc" "LastF1" "BestAcc" "BestF1" "Status"
    printf "%-10s  %-26s  %-8s  %-8s  %-8s  %-8s  %s\n" \
        "---------" "--------------------------" "--------" "--------" "--------" "--------" "------"
} | tee "${SUMMARY_TXT}"

for val_obj in "${OBJECTS[@]}"; do
    read -r -a train_objs <<< "$(train_objects_for_val "${val_obj}")"
    train_label_dirs="$(label_dirs_for_objects "${train_objs[@]}")"
    val_label_dirs="$(label_dirs_for_objects "${val_obj}")"
    train_objects_csv="$(IFS=+; echo "${train_objs[*]}")"

    train_roots_override="data.dataset.config.data_roots=[{data_path:${DATA_ROOT},label_dirs:{${train_label_dirs}}}]"
    train_data_path_override="data.dataset.data_path=${DATA_ROOT}"
    val_dataset_override="data.val_dataset={_target_:${DATASET_TARGET},config:{window_time:\${data.window_time},window_overlap:\${data.window_overlap},interpolating_freq:\${data.interpolating_freq},subtract_baseline:true,data_roots:[{data_path:${DATA_ROOT},label_dirs:{${val_label_dirs}}}]},data_path:${DATA_ROOT}}"
    run_name="angle_rope_grasp_objectwise_${EXP_NAME}_val_${val_obj}"
    log_file="${LOG_DIR}/val_${val_obj}.log"

    echo ""
    echo "================================================================"
    echo "  val object: ${val_obj}"
    echo "  train objects: ${train_objects_csv}"
    echo "  log: ${log_file}"
    echo "================================================================"

    cmd=(
        python train_task_brainco_angle.py
        "+experiment=${EXPERIMENT}"
        "${train_roots_override}"
        "${train_data_path_override}"
        "${val_dataset_override}"
        "experiment_name=${run_name}"
        "wandb.group=angle_rope_grasp_objectwise_${EXP_NAME}"
        "wandb.tags=[angle,grasp_prediction,objectwise,rope,val_${val_obj}]"
        "all_split=false"
        "${EXTRA_OVERRIDES[@]}"
    )

    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 "${cmd[@]}" 2>&1 | tee "${log_file}"; then
        echo "[OK] val=${val_obj}"
    else
        echo "[FAILED] val=${val_obj}"
        echo "OBJECTWISE_EXPERIMENT_FAILED" >> "${log_file}"
    fi

    result="$(extract_result "${log_file}")"
    IFS=',' read -r last_acc last_f1 best_acc best_f1 status <<< "${result}"
    printf "%-10s  %-26s  %-8s  %-8s  %-8s  %-8s  %s\n" \
        "${val_obj}" "${train_objects_csv}" "${last_acc:-n/a}" "${last_f1:-n/a}" \
        "${best_acc:-n/a}" "${best_f1:-n/a}" "${status}" | tee -a "${SUMMARY_TXT}"
    echo "${val_obj},${train_objects_csv},${last_acc},${last_f1},${best_acc},${best_f1},${status},${log_file}" >> "${SUMMARY_CSV}"
done

python3 - "${SUMMARY_CSV}" <<'PYEOF' | tee -a "${SUMMARY_TXT}"
import csv
import math
import statistics
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
ok = [r for r in rows if r["status"] == "OK"]

def values(key):
    out = []
    for row in ok:
        try:
            val = float(row[key])
        except (TypeError, ValueError):
            continue
        if not math.isnan(val):
            out.append(val)
    return out

print("")
print("Aggregate over held-out objects")
for prefix in ["last", "best"]:
    accs = values(f"{prefix}_acc")
    f1s = values(f"{prefix}_f1")
    if not accs or not f1s:
        print(f"  {prefix.title()}: n/a")
        continue
    acc_std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    f1_std = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0
    print(
        f"  {prefix.title()}: "
        f"MeanAcc={statistics.mean(accs):.4f} StdAcc={acc_std:.4f} "
        f"MeanF1={statistics.mean(f1s):.4f} StdF1={f1_std:.4f}"
    )
PYEOF

echo ""
echo "Done."
echo "  summary txt: ${SUMMARY_TXT}"
echo "  summary csv: ${SUMMARY_CSV}"
