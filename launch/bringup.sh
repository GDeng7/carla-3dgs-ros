#!/usr/bin/env bash
# Full bring-up: CARLA server + NuRec replay + ROS2/DDS
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/setup_env.sh"

CONFIG="${PROJECT_ROOT}/config/default.yaml"
USDZ="${1:-}"

echo "=== Step 1: Start CARLA server with native ROS2 DDS ==="
echo "In a separate terminal:"
echo "  cd \${CARLA_ROOT} && ./CarlaUE4.sh --ros2"
echo ""
read -r -p "Press Enter when CARLA is running..."

echo "=== Step 2: Start NuRec + replay ==="
ARGS=(--config "${CONFIG}")
if [[ -n "${USDZ}" ]]; then
  ARGS+=(--usdz "${USDZ}")
fi

python3 "${PROJECT_ROOT}/scripts/run_replay.py" "${ARGS[@]}"