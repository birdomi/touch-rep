#!/usr/bin/env bash
# Run XYZ object-wise grasp experiments across seeds for pretrained and scratch.
#
# Usage:
#   bash scripts/run_xyz_objectwise_multiseed.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_objectwise_multiseed.sh
#   SEEDS="0 1 2" RUN_PRETRAINED=0 bash scripts/run_xyz_objectwise_multiseed.sh
#   SEEDS="0 1 2" RUN_SCRATCH=0 bash scripts/run_xyz_objectwise_multiseed.sh
#   SEEDS="0 1 2" bash scripts/run_xyz_objectwise_multiseed.sh -- trainer.max_epochs=50
#
# Outputs one row per (model, seed, held-out object) and a mean ± std summary.

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

SEEDS_STRING="${SEEDS:-0 1 2}"
read -r -a SEED_LIST <<< "${SEEDS_STRING}"
RUN_PRETRAINED="${RUN_PRETRAINED:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
EXTRA_OVERRIDES=()

if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        echo "[ERROR] Extra Hydra overrides must follow --" >&2
        exit 1
    fi
    shift
    EXTRA_OVERRIDES=("$@")
fi

if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] SEEDS must contain at least one integer." >&2
    exit 1
fi
if [[ "${RUN_PRETRAINED}" != "1" && "${RUN_SCRATCH}" != "1" ]]; then
    echo "[ERROR] Enable at least one of RUN_PRETRAINED=1 or RUN_SCRATCH=1." >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_CSV="${SCRIPT_DIR}/results_xyz_objectwise_multiseed_${TIMESTAMP}.csv"
RESULT_MD="${SCRIPT_DIR}/results_xyz_objectwise_multiseed_${TIMESTAMP}.md"
MARKER_FILE=$(mktemp)
trap 'rm -f "${MARKER_FILE}"' EXIT

echo "model,seed,val_object,train_objects,last_acc,last_f1,best_acc,best_f1,status,source_csv,log_file" > "${RESULT_CSV}"

run_model() {
    local model_name="$1"
    local experiment="$2"
    local seed="$3"
    local experiment_name="${experiment##*/}"
    local source_csv

    touch "${MARKER_FILE}"
    echo ""
    echo "================================================================"
    echo "  model: ${model_name}  seed: ${seed}"
    echo "  experiment: ${experiment}"
    echo "================================================================"

    if ! bash scripts/run_xyz_objectwise_grasp.sh "${experiment}" -- "seed=${seed}" "${EXTRA_OVERRIDES[@]}"; then
        echo "[FAILED] ${model_name} seed=${seed}" >&2
    fi

    source_csv=$(find "${SCRIPT_DIR}" -maxdepth 1 -type f \
        -name "results_xyz_objectwise_${experiment_name}_*.csv" -newer "${MARKER_FILE}" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "${source_csv}" || ! -f "${source_csv}" ]]; then
        echo "[ERROR] Could not find summary CSV for ${model_name} seed=${seed}." >&2
        return 1
    fi

    python3 - "${RESULT_CSV}" "${source_csv}" "${model_name}" "${seed}" <<'PYEOF'
import csv
import sys

result_path, source_path, model, seed = sys.argv[1:]
with open(source_path, newline="") as source_file, open(result_path, "a", newline="") as result_file:
    reader = csv.DictReader(source_file)
    fields = [
        "model", "seed", "val_object", "train_objects", "last_acc", "last_f1",
        "best_acc", "best_f1", "status", "source_csv", "log_file",
    ]
    writer = csv.DictWriter(result_file, fieldnames=fields)
    for row in reader:
        writer.writerow({
            "model": model,
            "seed": seed,
            "val_object": row["val_object"],
            "train_objects": row["train_objects"],
            "last_acc": row["last_acc"],
            "last_f1": row["last_f1"],
            "best_acc": row["best_acc"],
            "best_f1": row["best_f1"],
            "status": row["status"],
            "source_csv": source_path,
            "log_file": row["log_file"],
        })
PYEOF
}

for seed in "${SEED_LIST[@]}"; do
    if [[ "${RUN_PRETRAINED}" == "1" ]]; then
        run_model "pretrained" "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope" "${seed}"
    fi
    if [[ "${RUN_SCRATCH}" == "1" ]]; then
        run_model "scratch" "brainco/ours_3d/task/grasp_prediction/dinov2_all_rope_scratch" "${seed}"
    fi
done

python3 - "${RESULT_CSV}" "${RESULT_MD}" <<'PYEOF'
import csv
import math
import statistics
import sys
from collections import defaultdict

csv_path, markdown_path = sys.argv[1:]
with open(csv_path, newline="") as file:
    rows = [row for row in csv.DictReader(file) if row["status"] == "OK"]

def mean_std(values):
    if not values:
        return "n/a", "n/a"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f}", f"{std:.4f}"

groups = defaultdict(list)
for row in rows:
    groups[(row["model"], row["val_object"])].append(row)

models = sorted({row["model"] for row in rows})
objects = sorted({row["val_object"] for row in rows})
seeds = sorted({row["seed"] for row in rows}, key=int)

lines = [
    "# XYZ DINOv2 Object-wise Multi-seed Comparison",
    "",
    f"Seeds: {', '.join(seeds)}",
    "",
    "Best validation epoch metrics, reported as mean ± sample standard deviation.",
    "",
    "| Held-out object | Model | n | Best Acc | Best F1 | Last Acc | Last F1 |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for obj in objects:
    for model in models:
        values = groups[(model, obj)]
        best_acc, best_acc_std = mean_std([float(r["best_acc"]) for r in values])
        best_f1, best_f1_std = mean_std([float(r["best_f1"]) for r in values])
        last_acc, last_acc_std = mean_std([float(r["last_acc"]) for r in values])
        last_f1, last_f1_std = mean_std([float(r["last_f1"]) for r in values])
        lines.append(
            f"| {obj} | {model} | {len(values)} | {best_acc} ± {best_acc_std} | "
            f"{best_f1} ± {best_f1_std} | {last_acc} ± {last_acc_std} | "
            f"{last_f1} ± {last_f1_std} |"
        )

lines.extend([
    "",
    "## Overall",
    "",
    "| Model | n object-seed runs | Best Acc | Best F1 | Last Acc | Last F1 |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
])
for model in models:
    values = [row for row in rows if row["model"] == model]
    metrics = []
    for key in ("best_acc", "best_f1", "last_acc", "last_f1"):
        mean, std = mean_std([float(row[key]) for row in values])
        metrics.append(f"{mean} ± {std}")
    lines.append(f"| {model} | {len(values)} | " + " | ".join(metrics) + " |")

with open(markdown_path, "w") as file:
    file.write("\n".join(lines) + "\n")
PYEOF

echo ""
echo "Done."
echo "  Per-run metrics: ${RESULT_CSV}"
echo "  Aggregated report: ${RESULT_MD}"
