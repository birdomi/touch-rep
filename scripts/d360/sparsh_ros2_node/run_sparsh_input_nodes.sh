#!/bin/bash
set -eo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
DEVICES=("$@")

# Validate core dependency
if ! command -v /usr/bin/env python3 >/dev/null; then
    echo "Python3 not found!"
    exit 1
fi

for device in "${DEVICES[@]}"; do
    script_path="${REPO_ROOT}/scripts/d360/eval_plug_insertion/sparsh_d360_input_node.py"
    log_file="/tmp/${device}_obs_node.log"
    pid_file="/tmp/${device}_obs_node.pid"

    # Cleanup previous runs
    rm -f "${pid_file}"

    # Validate script exists
    if [ ! -f "${script_path}" ]; then
        echo "Error: Script not found at ${script_path}" >&2
        exit 1
    fi

    # Launch with full path to python
    /usr/bin/env python3 "${script_path}" --device "${device}" >"${log_file}" 2>&1 &
    pid=$!
    echo "${pid}" >"${pid_file}"
    echo "Launched ${device} with PID ${pid}"

    # Verify process actually started
    sleep 0.5
    if ! ps -p "${pid}" >/dev/null; then
        echo "Process ${pid} died immediately! Check ${log_file}" >&2
        exit 1
    fi
done

cleanup() {
    trap '' SIGINT SIGTERM
    echo "Cleaning up..."
    for device in "${DEVICES[@]}"; do
        pid_file="/tmp/${device}_obs_node.pid"
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            echo "Killing ${device}_obs_node.py (PID ${pid})"
            kill -9 "${pid}" 2>/dev/null || true
            rm -f "${pid_file}"
        fi
    done
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep alive with process monitoring
while true; do
    for device in "${DEVICES[@]}"; do
        pid=$(cat "/tmp/${device}_obs_node.pid")
        if ! ps -p "${pid}" >/dev/null; then
            echo "Process ${pid} (${device}) died unexpectedly!" >&2
            cleanup
        fi
    done
    sleep 2
done
