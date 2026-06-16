#!/usr/bin/env bash
# Environment setup for CARLA + NuRec + ROS2 on Ubuntu 22.04
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${CARLA_ROOT:-}" ]]; then
  echo "ERROR: Set CARLA_ROOT to your CARLA package directory."
  echo "  export CARLA_ROOT=/path/to/CARLA"
  exit 1
fi

# Python virtual environment
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  python3 -m venv "${PROJECT_ROOT}/.venv"
fi
source "${PROJECT_ROOT}/.venv/bin/activate"
pip install -U pip
pip install -r "${PROJECT_ROOT}/requirements.txt"

# CARLA Python API
WHEEL="$(ls "${CARLA_ROOT}"/PythonAPI/carla/dist/carla-*-py3*.whl 2>/dev/null | head -1 || true)"
if [[ -n "${WHEEL}" ]]; then
  pip install "${WHEEL}"
else
  echo "WARN: CARLA wheel not found under ${CARLA_ROOT}/PythonAPI/carla/dist/"
fi

# NuRec installer (downloads Docker image + sample data)
NUREC_INSTALLER="${CARLA_ROOT}/PythonAPI/examples/nvidia/nurec/install_nurec.sh"
if [[ -f "${NUREC_INSTALLER}" ]]; then
  echo "Run NuRec installer manually if not done yet:"
  echo "  cd ${CARLA_ROOT}/PythonAPI/examples/nvidia/nurec && ./install_nurec.sh"
fi

# NuRec Docker image
export NUREC_IMAGE="${NUREC_IMAGE:-docker.io/carlasimulator/nvidia-nurec-grpc:0.2.0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ROS2 (source your distro — Humble recommended)
if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  source /opt/ros/humble/setup.bash
  echo "Sourced ROS2 Humble."
elif [[ -f "/opt/ros/foxy/setup.bash" ]]; then
  source /opt/ros/foxy/setup.bash
  echo "Sourced ROS2 Foxy."
else
  echo "WARN: ROS2 not found. Install ROS2 Humble for DDS bridge."
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "Environment ready. Project: ${PROJECT_ROOT}"