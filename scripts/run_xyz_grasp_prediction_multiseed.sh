#!/usr/bin/env bash
# Run BrainCo XYZ grasp-prediction pretrained/scratch comparisons over seeds.
# Both evaluation protocols run by default:
#   - ID: episode-level K-fold splits using all objects.
#   - OOD: leave-one-object-out (object-wise) splits.
#
# Usage:
#   bash scripts/run_xyz_grasp_prediction_multiseed.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_grasp_prediction_multiseed.sh
#   RUN_OBJECTWISE=0 bash scripts/run_xyz_grasp_prediction_multiseed.sh
#   RUN_ID=0 bash scripts/run_xyz_grasp_prediction_multiseed.sh
#   SEEDS="0 1 2" bash scripts/run_xyz_grasp_prediction_multiseed.sh -- trainer.max_epochs=50

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PRETRAINED_EXPERIMENT="${PRETRAINED_EXPERIMENT:-brainco/ours_3d/task/grasp_prediction/dinov2_all_rope}"
SCRATCH_EXPERIMENT="${SCRATCH_EXPERIMENT:-brainco/ours_3d/task/grasp_prediction/dinov2_all_rope_scratch}"
SEEDS_STRING="${SEEDS:-0 1 2}"
read -r -a SEED_LIST <<< "${SEEDS_STRING}"
SPLIT_SEED="${SPLIT_SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-4}"
RUN_PRETRAINED="${RUN_PRETRAINED:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_ID="${RUN_ID:-1}"
RUN_OBJECTWISE="${RUN_OBJECTWISE:-1}"
EXTRA_OVERRIDES=()

if [[ $# -gt 0 ]]; then
    [[ "$1" == "--" ]] || { echo "Usage: $0 [-- key=value ...]" >&2; exit 1; }
    shift
    EXTRA_OVERRIDES=("$@")
fi

[[ ${#SEED_LIST[@]} -gt 0 ]] || { echo "[ERROR] SEEDS must contain at least one integer." >&2; exit 1; }
[[ "${RUN_PRETRAINED}" == 1 || "${RUN_SCRATCH}" == 1 ]] || { echo "[ERROR] Enable RUN_PRETRAINED or RUN_SCRATCH." >&2; exit 1; }
[[ "${RUN_ID}" == 1 || "${RUN_OBJECTWISE}" == 1 ]] || { echo "[ERROR] Enable RUN_ID or RUN_OBJECTWISE." >&2; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/xyz_grasp_prediction_id_multiseed_${TIMESTAMP}"
ID_CSV="${SCRIPT_DIR}/results_xyz_grasp_prediction_id_multiseed_${TIMESTAMP}.csv"
ID_MD="${SCRIPT_DIR}/results_xyz_grasp_prediction_id_multiseed_${TIMESTAMP}.md"
mkdir -p "${LOG_DIR}"

extract_result() {
    python3 - "$1" <<'PYEOF'
import re, sys
text = open(sys.argv[1]).read()
if "ID_EXPERIMENT_FAILED" in text:
    print(",,,,FAILED"); raise SystemExit
def metric(label):
    m = re.search(rf"K-FOLD SUMMARY \({label} Epoch\).*?^\s*Mean\s+([\d.]+|nan)\s+([\d.]+|nan)", text, re.DOTALL | re.MULTILINE)
    return m.groups() if m else ("", "")
last_acc, last_f1 = metric("Last")
best_acc, best_f1 = metric("Best")
print(",".join((last_acc, last_f1, best_acc, best_f1, "OK" if any((last_acc,last_f1,best_acc,best_f1)) else "INCOMPLETE")))
PYEOF
}

run_id() {
    local model="$1" experiment="$2" seed="$3"
    local exp_name="${experiment##*/}"
    local log_file="${LOG_DIR}/${model}_seed_${seed}.log"
    local run_name="xyz_grasp_id_${exp_name}_${model}_seed_${seed}"
    local cmd=(python train_task_brainco_angle.py "+experiment=${experiment}" "seed=${seed}" "+split_seed=${SPLIT_SEED}" "experiment_name=${run_name}" "wandb.group=xyz_grasp_id_${exp_name}_${model}_multiseed" "wandb.tags=[brainco,xyz,dinov2,grasp_prediction,id,${model},multiseed,seed_${seed}]" "${EXTRA_OVERRIDES[@]}" --all_split --num_folds "${NUM_FOLDS}")
    echo "[ID] model=${model}, seed=${seed}"
    if XFORMERS_DISABLED=TRUE HYDRA_FULL_ERROR=1 "${cmd[@]}" 2>&1 | tee "${log_file}"; then status=OK; else status=FAILED; echo ID_EXPERIMENT_FAILED >> "${log_file}"; fi
    IFS=',' read -r last_acc last_f1 best_acc best_f1 parsed_status <<< "$(extract_result "${log_file}")"
    [[ "${status}" == OK ]] || parsed_status=FAILED
    echo "${model},${seed},${last_acc},${last_f1},${best_acc},${best_f1},${parsed_status},${log_file}" >> "${ID_CSV}"
}

echo "XYZ grasp-prediction multi-seed: seeds=${SEEDS_STRING}; ID=${RUN_ID}; object-wise=${RUN_OBJECTWISE}"
if [[ "${RUN_ID}" == 1 ]]; then
    echo "model,seed,last_acc,last_f1,best_acc,best_f1,status,log_file" > "${ID_CSV}"
    for seed in "${SEED_LIST[@]}"; do
        [[ "${RUN_PRETRAINED}" == 1 ]] && run_id pretrained "${PRETRAINED_EXPERIMENT}" "${seed}"
        [[ "${RUN_SCRATCH}" == 1 ]] && run_id scratch "${SCRATCH_EXPERIMENT}" "${seed}"
    done
    python3 - "${ID_CSV}" "${ID_MD}" <<'PYEOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
lines = ["# XYZ Grasp Prediction: ID Multi-seed Results", "", "| Model | Seed | Last Acc | Last F1 | Best Acc | Best F1 | Status |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
for r in rows: lines.append(f"| {r['model']} | {r['seed']} | {r['last_acc'] or 'n/a'} | {r['last_f1'] or 'n/a'} | {r['best_acc'] or 'n/a'} | {r['best_f1'] or 'n/a'} | {r['status']} |")
lines += ["", "## Mean ± sample standard deviation", "", "| Model | Last Acc | Last F1 | Best Acc | Best F1 |", "| --- | ---: | ---: | ---: | ---: |"]
for model in sorted({r['model'] for r in rows}):
    def stat(key):
        vals = [float(r[key]) for r in rows if r['model'] == model and r['status'] == 'OK' and r[key] not in ('', 'nan')]
        return 'n/a' if not vals else f"{statistics.mean(vals):.4f} ± {(statistics.stdev(vals) if len(vals)>1 else 0):.4f}"
    lines.append(f"| {model} | {stat('last_acc')} | {stat('last_f1')} | {stat('best_acc')} | {stat('best_f1')} |")
open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')
PYEOF
    echo "ID results: ${ID_MD}"
fi

if [[ "${RUN_OBJECTWISE}" == 1 ]]; then
    SEEDS="${SEEDS_STRING}" RUN_PRETRAINED="${RUN_PRETRAINED}" RUN_SCRATCH="${RUN_SCRATCH}" bash scripts/run_xyz_objectwise_multiseed.sh -- "${EXTRA_OVERRIDES[@]}"
fi
