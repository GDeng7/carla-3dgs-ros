# Architecture Comparison

## Two Approaches

| | **NuRec mode** (what we built first) | **Co-simulation** (Grok / SplatSim) |
|---|---|---|
| **Rendering** | NVIDIA NuRec Docker (closed gRPC) | Your own gsplat engine |
| **CARLA role** | Replay + actors + NuRec frames bundled | Physics, traffic, pose only |
| **Control** | Limited to NuRec API | Full pipeline ownership |
| **Data format** | NuRec `.usdz` | PLY / glTF Gaussian Splatting tileset |
| **ROS2** | CARLA `--ros2` + optional bridge | Renderer publishes images via DDS |
| **Best for** | Quick demo with NVIDIA dataset | DriveArena / World Simulator lab |

## NuRec Mode (Integrated Replay)

```
ROS2  ←──DDS──→  CARLA (--ros2)  ←──API──→  Replay Script
                                                  │
                                            NuRec Docker (gRPC)
                                                  │
                                            NuRec .usdz scene
```

- Uses CARLA's official `example_nurec_replay_save_images.py`
- Rendering locked inside NVIDIA's Docker container
- Good for validating with [Physical AI NuRec dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)

**Run:** `python scripts/run_replay.py`

## Co-simulation Mode (Recommended for Your Lab)

```
┌─────────────────┐
│  CARLA Server   │  physics, traffic, actor behaviour
└────────┬────────┘
         │ camera pose each tick
         ▼
┌─────────────────┐
│ 3DGS Renderer   │  gsplat (self-owned, like SplatSim)
└────────┬────────┘
         │ sensor_msgs/Image
         ▼
┌─────────────────┐
│  ROS2 / DDS     │  → perception / Autoware
└─────────────────┘
```

This matches [tier4/splatsim](https://github.com/tier4/splatsim):

- CARLA simulates traffic participants
- SplatSim renders camera images from 3DGS
- CycloneDDS publishes directly to ROS 2 / Autoware

**Run:** `python scripts/run_cosim.py --config config/cosim.yaml`

## Data Format Note

Your sample file is **NuRec USDZ** (NVIDIA format with `volume.nurec`, `checkpoint.ckpt`).

SplatSim uses **Cesium glTF Gaussian Splatting** tilesets + gsplat PLY.

To use co-simulation with a self-created map:

1. Train with [3DGUT / gsplat](https://github.com/nv-tlabs/3dgrut)
2. Export as **PLY** (gsplat-compatible) or glTF splat extension
3. Point `config/cosim.yaml` → `renderer.scene_path` at your export

## RTX 5090 / Blackwell — use co-simulation only

NuRec Docker 0.2 does **not** support Blackwell (RTX 5090). Co-simulation avoids NuRec entirely:

| Component | NuRec mode | Co-simulation |
|-----------|------------|---------------|
| Rendering GPU stack | NuRec Docker (broken on 5090) | PyTorch + gsplat (cu128+) |
| Scene format | `.usdz` (NuRec proprietary) | Standard `.ply` (self-built) |
| CARLA role | Bundled replay | Physics + traffic only |

Install for 5090:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-cosim.txt
```

## Recommendation

| Goal | Use |
|------|-----|
| RTX 5090 without NuRec | **`run_cosim.py`** |
| Self-built 3DGS maps | **`run_cosim.py`** + PLY export |
| Quick test with NVIDIA NuRec USDZ only | `run_replay.py` (needs supported GPU) |
| Production Autoware lab | [SplatSim](https://github.com/tier4/splatsim) |