# CARLA + 3DGS + ROS2/DDS

Python toolkit for rendering **3D Gaussian Splatting (3DGS) maps** in **CARLA** with **ROS2/DDS** — supporting two architectures:

| Mode | Command | When to use |
|------|---------|-------------|
| **Co-simulation (Linux)** | `python scripts/run_cosim.py` | CARLA + gsplat on RTX 5090 — production lab |
| **Co-simulation (macOS)** | `python scripts/run_cosim_macos.py` | Develop/test on Mac without CARLA |
| **NuRec integrated** | `python scripts/run_replay.py` | Quick start with NVIDIA NuRec Docker + sample USDZ |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full comparison.

Based on:

- [CARLA 0.9.16 NuRec integration](https://carla.org/2025/06/11/release-0.9.16-pre/)
- [CARLA NuRec documentation](https://carla.readthedocs.io/en/latest/nvidia_nurec/)
- [NVIDIA: Render real-world scenes in simulation](https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/)

## Architecture

### Co-simulation (recommended for your lab — matches Grok / SplatSim)

```
CARLA (physics + traffic)  →  camera pose  →  3DGS renderer (gsplat)  →  ROS2/DDS  →  perception
```

Reference: [tier4/splatsim](https://github.com/tier4/splatsim)

### NuRec mode (what we built first — different from Grok's recommendation)

```
CARLA + NuRec Docker (gRPC)  →  bundled replay  →  ROS2
```

**Data flow:**

1. Your **USDZ** file (trained with [3DGUT](https://github.com/nv-tlabs/3dgrut) / NuRec) holds the neural scene.
2. **NuRec gRPC** container renders photorealistic views from arbitrary camera poses.
3. **CARLA** replays actors, map (OpenDRIVE), and trajectories from the same scene.
4. **ROS2/DDS** carries camera images, odometry, and control — either via CARLA native `--ros2` or this project's Python bridge.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Ubuntu 22.04** | CARLA + NuRec officially support Linux |
| **NVIDIA GPU + CUDA 12.8+** | Required for NuRec rendering |
| **Docker + NVIDIA Container Toolkit** | NuRec runs in Docker |
| **CARLA 0.9.16 nightly** | [Download](https://carla.readthedocs.io/en/latest/download/) |
| **Python 3.10** | 3.11+ not supported by NuRec installer |
| **ROS2 Humble** (recommended) | For DDS bridge |

> **Note:** CARLA and NuRec do not run on macOS. Use `run_cosim_macos.py` on Mac for development; use `run_cosim.py` on Linux with your RTX 5090 for full co-simulation.

## macOS co-simulation (no CARLA)

Develop the pose → render → ROS pipeline on your Mac:

```bash
pip install -r requirements-macos.txt
python scripts/run_cosim_macos.py --config config/cosim_macos.yaml
```

| Component | macOS replacement |
|-----------|-------------------|
| CARLA physics | Synthetic pose simulator (`straight`, `circle`, or `replay` JSON) |
| gsplat (CUDA) | CPU `preview` renderer (projects your PLY Gaussians) |
| ROS2 | Optional (`--ros` if ROS2 Humble installed via Homebrew) |

**Workflow:** Record ego poses on Linux with CARLA → export JSON → replay on Mac with `trajectory: replay`.

```bash
# Generate a pose template
python scripts/run_cosim_macos.py --export-pose-template assets/poses/my_recording.json

# Save rendered frames (no display needed)
python scripts/run_cosim_macos.py --save-frames --no-preview
```

Press `q` in the preview window to stop.

## Quick start

### 1. Install CARLA + NuRec

```bash
export CARLA_ROOT=/path/to/CARLA
cd $CARLA_ROOT/PythonAPI/examples/nvidia/nurec
./install_nurec.sh   # needs HuggingFace token for sample data
```

### 2. Set up this project

```bash
cd carla-3dgs-ros
source scripts/setup_env.sh
```

Edit `config/default.yaml` and set your USDZ path:

```yaml
scene:
  usdz_path: "/path/to/your_scene.usdz"
```

### 3. Inspect your scene (works on macOS too)

```bash
python scripts/inspect_scene.py
python scripts/run_replay.py --dry-run
```

### 4. Run full replay (Linux + GPU)

**Terminal 1 — CARLA with ROS2:**

```bash
cd $CARLA_ROOT
./CarlaUE4.sh --ros2
```

**Terminal 2 — Replay:**

```bash
source scripts/setup_env.sh
python scripts/run_replay.py
```

Or use the bring-up helper:

```bash
./launch/bringup.sh /path/to/your_scene.usdz
```

## ROS2 topics

With CARLA launched using `--ros2`, sensors publish on DDS automatically:

| Topic | Message |
|-------|---------|
| `/carla/ego/front_wide_120fov/image` | `sensor_msgs/Image` |
| `/carla/ego/odometry` | `nav_msgs/Odometry` |
| `/carla/ego/vehicle_control_cmd` | `CarlaEgoVehicleControl` |

Install [ros-carla-msgs](https://github.com/carla-simulator/ros-carla-msgs) for control messages.

Verify with:

```bash
ros2 topic list
ros2 topic echo /carla/ego/front_wide_120fov/image --no-arr
```

## Project layout

```
carla-3dgs-ros/
├── config/
│   ├── default.yaml      # Main settings
│   └── cameras.yaml      # 6-camera AV rig
├── carla_3dgs_ros/
│   ├── scene.py          # USDZ metadata loader
│   ├── nurec_client.py   # NuRec Docker lifecycle
│   ├── carla_bridge.py   # CARLA API wrapper
│   ├── ros_bridge.py     # ROS2/DDS publishers
│   └── replay.py         # Main orchestrator
├── scripts/
│   ├── run_replay.py
│   ├── inspect_scene.py
│   └── setup_env.sh
└── launch/
    └── bringup.sh
```

## Creating your own 3DGS map

1. Capture multi-camera driving data (or use [Physical AI dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)).
2. Train with [3DGUT](https://github.com/nv-tlabs/3dgrut) and export USDZ:

   ```bash
   python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
       export_usdz.enabled=true \
       export_usdz.apply_normalizing_transform=true
   ```

3. Point `config/default.yaml` → `scene.usdz_path` at your exported `.usdz`.
4. Run replay as above.

## CLI reference

```bash
python scripts/run_replay.py --help

  --config PATH       YAML config (default: config/default.yaml)
  --usdz PATH         Override scene file
  --dry-run           Validate only
  --integrated        Skip CARLA official script
  --no-ros            Disable Python ROS2 bridge
  --no-nurec-container  Assume NuRec is already running
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `carla module not found` | `pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-*.whl` |
| NuRec Docker fails | Check `docker run --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |
| No ROS topics | Launch CARLA with `--ros2`; call `sensor.enable_for_ros()` |
| Black NuRec frames | Ensure USDZ contains `volume.nurec` and `checkpoint.ckpt` |

## License

This project code is provided as-is. NuRec dataset and USDZ scenes are subject to the [NVIDIA AV Dataset License](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/blob/main/LICENSE.pdf).