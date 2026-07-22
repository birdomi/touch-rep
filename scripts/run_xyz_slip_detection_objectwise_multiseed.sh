#!/usr/bin/env bash
# Leave-one-object-out (OOD object) XYZ slip-detection experiments across seeds.
#
# Usage:
#   bash scripts/run_xyz_slip_detection_objectwise_multiseed.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_slip_detection_objectwise_multiseed.sh
#   SEEDS="0 1" RUN_SCRATCH=0 bash scripts/run_xyz_slip_detection_objectwise_multiseed.sh
#   bash scripts/run_xyz_slip_detection_objectwise_multiseed.sh -- trainer.max_epochs=50

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT:-brainco/ours_3d/task/slip_detection/dinov2_all_rope}"
SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT:-brainco/ours_3d/task/slip_detection/dinov2_all_rope_scratch}"
SEEDS_STRING="${SEEDS:-0 1 2}"
read -r -a SEED_LIST <<< "${SEEDS_STRING}"
RUN_PRETRAINED="${RUN_PRETRAINED:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
OBJECTS_STRING="${OBJECTS:-slip_doll slip_evacase slip_pot}"
read -r -a OBJECT_LIST <<< "${OBJECTS_STRING}"
EXTRA_OVERRIDES=()

if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        echo "[ERROR] Extra Hydra overrides must follow --" >&2
        exit 1
    fi
    shift
    EXTRA_OVERRIDES=("$@")
fi
if [[ ${#SEED_LIST[@]} -eq 0 || ${#OBJECT_LIST[@]} -lt 2 ]]; then
    echo "[ERROR] Set at least one seed and at least two OBJECTS." >&2
    exit 1
fi
if [[ "${RUN_PRETRAINED}" != "1" && "${RUN_SCRATCH}" != "1" ]]; then
    echo "[ERROR] Enable RUN_PRETRAINED=1 and/or RUN_SCRATCH=1." >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/xyz_slip_detection_objectwise_${TIMESTAMP}"
RESULT_CSV="${SCRIPT_DIR}/results_xyz_slip_detection_objectwise_multiseed_${TIMESTAMP}.csv"
RESULT_MD="${SCRIPT_DIR}/results_xyz_slip_detection_objectwise_multiseed_${TIMESTAMP}.md"
mkdir -p "${LOG_DIR}"
echo "model,seed,val_object,train_objects,last_acc,last_f1,best_acc,best_f1,status,log_file" > "${RESULT_CSV}"

classes_override() {
    local result="[" obj
    for obj in "$@"; do
        [[ "${result}" != "[" ]] && result+=","
        result+="${obj}"
    done
    echo "${result}]"
}

extract_result() {
    local log_file="$1"
    python3 - "${log_file}" <<'PYEOF'
import re
import sys
text = open(sys.argv[1]).read() if __import__('os').path.exists(sys.argv[1]) else ''
if 'SLIP_OBJECTWISE_EXPERIMENT_FAILED' in text:
    print(',,,,FAILED'); raise SystemExit
def metric(label):
    match = re.search(rf'{label}\s+[^A-Za-z0-9]*\s+Acc:\s+([\d.]+|nan)\s+F1:\s+([\d.]+|nan)', text)
    return match.groups() if match else ('', '')
last_acc, last_f1 = metric('Last')
best_acc, best_f1 = metric('Best')
print(','.join((last_acc, last_f1, best_acc, best_f1, 'OK' if any((last_acc,last_f1,best_acc,best_f1)) else 'INCOMPLETE')))
PYEOF
}

run_experiment() {
    local model="$1" experiment="$2" seed="$3" val_object="$4"
    local train_objects=() object
    for object in "${OBJECT_LIST[@]}"; do [[ "${object}" != "${val_object}" ]] && train_objects+=("${object}"); done
    local train_classes val_classes train_csv exp_name log_file run_name status result
    train_classes="$(classes_override "${train_objects[@]}")"
    val_classes="$(classes_override "${val_object}")"
    train_csv="$(IFS=+; echo "${train_objects[*]}")"
    exp_name="${experiment##*/}"
    log_file="${LOG_DIR}/${model}_seed_${seed}_val_${val_object}.log"
    run_name="xyz_slip_detection_objectwise_${exp_name}_${model}_seed_${seed}_val_${val_object}"
    cmd=(python train_task_brainco_angle.py "+experiment=${experiment}"
        "seed=${seed}" "data.dataset.config.classes=${train_classes}"
        "data.val_dataset={_target_:tactile_ssl.data.brainco_xyz_slip_detection_dataset.BraincoXYZSlipDetectionDataset,config:{window_time:\${data.window_time},window_overlap:\${data.window_overlap},interpolating_freq:\${data.interpolating_freq},subtract_baseline:true,align_xyz_to_npz:true,input_window_frames:\${data.dataset.config.input_window_frames},input_window_stride:\${data.dataset.config.input_window_stride},exclude_before_slip_start_frames:\${data.dataset.config.exclude_before_slip_start_frames},exclude_after_slip_end_frames:\${data.dataset.config.exclude_after_slip_end_frames},classes:${val_classes}},data_path:\${data.dataset.data_path},brainco_urdf_path:\${data.dataset.brainco_urdf_path}}"
        "experiment_name=${run_name}" "wandb.group=xyz_slip_detection_objectwise_${exp_name}_${model}"
        "wandb.tags=[brainco,xyz,dinov2,slip_detection,objectwise,ood,${model},seed_${seed},val_${val_object}]"
        "all_split=false" "${EXTRA_OVERRIDES[@]}")
    echo "Running ${model}, seed=${seed}, held-out=${val_object} (train: ${train_csv})"
    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 "${cmd[@]}" 2>&1 | tee "${log_file}"; then status=OK; else status=FAILED; echo "SLIP_OBJECTWISE_EXPERIMENT_FAILED" >> "${log_file}"; fi
    result="$(extract_result "${log_file}")"
    IFS=',' read -r last_acc last_f1 best_acc best_f1 parsed_status <<< "${result}"
    [[ "${status}" == OK ]] || parsed_status=FAILED
    echo "${model},${seed},${val_object},${train_csv},${last_acc},${last_f1},${best_acc},${best_f1},${parsed_status},${log_file}" >> "${RESULT_CSV}"
}

for seed in "${SEED_LIST[@]}"; do
    for val_object in "${OBJECT_LIST[@]}"; do
        [[ "${RUN_PRETRAINED}" == 1 ]] && run_experiment pretrained "${PRETRAINED_EXPERIMENT}" "${seed}" "${val_object}"
        [[ "${RUN_SCRATCH}" == 1 ]] && run_experiment scratch "${SCRATCH_EXPERIMENT}" "${seed}" "${val_object}"
    done
done

python3 - "${RESULT_CSV}" "${RESULT_MD}" <<'PYEOF'
import csv, math, statistics, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
groups = defaultdict(list)
for row in rows:
    if row['status'] == 'OK': groups[(row['val_object'], row['model'])].append(row)
lines = ['# XYZ Slip-detection Object-wise Multi-seed Results', '', '| Held-out object | Model | n | Best Acc | Best F1 | Last Acc | Last F1 |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for (obj, model), values in sorted(groups.items()):
    def summary(key):
        xs = [float(r[key]) for r in values if r[key] and not math.isnan(float(r[key]))]
        return 'n/a' if not xs else f'{statistics.mean(xs):.4f} ± {(statistics.stdev(xs) if len(xs)>1 else 0):.4f}'
    lines.append(f'| {obj} | {model} | {len(values)} | {summary("best_acc")} | {summary("best_f1")} | {summary("last_acc")} | {summary("last_f1")} |')
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')
PYEOF

echo "Done. Per-run metrics: ${RESULT_CSV}"
echo "Aggregated report: ${RESULT_MD}"
