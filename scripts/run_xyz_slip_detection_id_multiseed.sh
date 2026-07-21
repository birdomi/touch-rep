#!/usr/bin/env bash
# In-distribution (all-object) XYZ slip-detection experiments across seeds.
#
# This is the named ID entrypoint. It delegates to the existing all-object
# episode-level K-fold runner, which uses all configured objects.
#
# Usage:
#   bash scripts/run_xyz_slip_detection_id_multiseed.sh
#   SEEDS="0 1 2 3 4" bash scripts/run_xyz_slip_detection_id_multiseed.sh
#   SEEDS="0 1 2" RUN_SCRATCH=0 bash scripts/run_xyz_slip_detection_id_multiseed.sh
#   bash scripts/run_xyz_slip_detection_id_multiseed.sh -- trainer.max_epochs=50

[ -z "${BASH_VERSION:-}" ] && exec bash "$0" "$@"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_xyz_slip_detection_multiseed.sh" "$@"
