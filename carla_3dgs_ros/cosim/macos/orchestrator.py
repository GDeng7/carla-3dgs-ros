"""macOS co-simulation orchestrator — synthetic poses + renderer + optional ROS2."""

from __future__ import annotations

import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from carla_3dgs_ros.config_loader import load_config
from carla_3dgs_ros.cosim.gsplat_renderer import SceneAlignment, create_renderer
from carla_3dgs_ros.cosim.macos.pose_simulator import (
    PoseSimulator,
    PoseSimulatorConfig,
    export_poses_template,
)
from carla_3dgs_ros.cosim.macos.preview_renderer import PreviewRenderer
from carla_3dgs_ros.cosim.pose import CameraMount
from carla_3dgs_ros.cosim.renderer_base import RendererConfig
from carla_3dgs_ros.ros_bridge import RosBridge, RosBridgeConfig
from carla_3dgs_ros.utils import ensure_directory

logger = logging.getLogger(__name__)


@dataclass
class MacCoSimOptions:
    config_path: str
    enable_ros: bool = False
    show_preview: bool = True
    save_frames: bool = False


class MacCoSimOrchestrator:
    """
    macOS co-simulation (CARLA-free).

        Pose Simulator (synthetic or replay JSON)
                ↓ camera pose
        3DGS Renderer (preview | mock | gsplat*)
                ↓ image
        ROS2 bridge (optional) / OpenCV preview / save frames

    * gsplat requires CUDA — not available on Mac. Use preview or mock.

    Record poses on Linux with CARLA, replay on Mac with trajectory=replay.
    """

    def __init__(self, options: MacCoSimOptions) -> None:
        self.options = options
        self.config = load_config(options.config_path)

    @staticmethod
    def check_platform() -> None:
        if platform.system() != "Darwin":
            logger.warning(
                "run_cosim_macos is intended for macOS. "
                "On Linux with CARLA, use: python scripts/run_cosim.py"
            )

    def _create_renderer(self, renderer_cfg: dict, alignment: SceneAlignment):
        backend = str(renderer_cfg.get("backend", "preview"))
        device = str(renderer_cfg.get("device", "cpu"))

        if backend == "gsplat" and device == "cuda":
            logger.warning(
                "gsplat+cuda is not available on macOS. "
                "Falling back to preview renderer."
            )
            backend = "preview"

        cfg = RendererConfig(
            scene_path=renderer_cfg["scene_path"],
            width=int(renderer_cfg["width"]),
            height=int(renderer_cfg["height"]),
            fov=float(renderer_cfg["fov"]),
            device=device,
            near_plane=float(renderer_cfg.get("near_plane", 0.01)),
            far_plane=float(renderer_cfg.get("far_plane", 1000.0)),
            background_color=tuple(renderer_cfg.get("background_color", [0, 0, 0])),
            backend=backend,
        )

        if backend == "preview":
            return PreviewRenderer(cfg, alignment=alignment)
        return create_renderer(cfg, alignment=alignment)

    def _start_ros(self) -> RosBridge | None:
        if not self.options.enable_ros:
            return None
        if not RosBridge.available():
            logger.warning("rclpy not installed — skipping ROS2. Install via: brew install ros-humble-desktop")
            return None

        ros_cfg = self.config["ros2"]
        topics = dict(ros_cfg.get("topics", {}))
        topics.setdefault("front_wide_image", topics.pop("image", "camera/front/image_raw"))
        bridge = RosBridge(
            RosBridgeConfig(
                namespace=ros_cfg["namespace"],
                ego_name=ros_cfg["ego_name"],
                frame_id=ros_cfg.get("frame_id", "front_camera"),
                publish_rate_hz=int(ros_cfg.get("publish_rate_hz", 20)),
                topics=topics,
            )
        )
        bridge.initialize()
        return bridge

    def run(self) -> int:
        self.check_platform()

        sim_cfg = self.config["simulator"]
        renderer_cfg = self.config["renderer"]
        camera_cfg = self.config["camera"]
        output_cfg = self.config.get("output", {})
        align_cfg = self.config.get("alignment", {})

        alignment = SceneAlignment(
            offset=tuple(align_cfg.get("offset", [0.0, 0.0, 0.0])),
            yaw_offset_deg=float(align_cfg.get("yaw_offset_deg", 0.0)),
        )

        mount = CameraMount(
            position=tuple(camera_cfg["position"]),
            rotation=tuple(camera_cfg["rotation"]),
        )

        simulator = PoseSimulator(
            PoseSimulatorConfig(
                trajectory=sim_cfg.get("trajectory", "straight"),
                speed_mps=float(sim_cfg.get("speed_mps", 10.0)),
                duration_sec=float(sim_cfg.get("duration_sec", 10.0)),
                tick_hz=float(sim_cfg.get("tick_hz", 20.0)),
                start_position=tuple(sim_cfg.get("start_position", [0.0, 0.0, 0.0])),
                start_yaw_deg=float(sim_cfg.get("start_yaw_deg", 0.0)),
                circle_radius_m=float(sim_cfg.get("circle_radius_m", 30.0)),
                replay_file=sim_cfg.get("replay_file", ""),
            ),
            mount=mount,
        )

        renderer = self._create_renderer(renderer_cfg, alignment)
        if renderer_cfg.get("backend") != "mock":
            try:
                renderer.load_scene()
            except FileNotFoundError as exc:
                logger.warning("%s — switching to mock renderer.", exc)
                renderer = create_renderer(
                    RendererConfig(
                        scene_path="",
                        width=int(renderer_cfg["width"]),
                        height=int(renderer_cfg["height"]),
                        fov=float(renderer_cfg["fov"]),
                        backend="mock",
                    )
                )
                renderer.load_scene()
        else:
            renderer.load_scene()

        ros = self._start_ros()
        preview = self._open_preview() if self.options.show_preview else None

        output_dir = None
        if self.options.save_frames or output_cfg.get("save_frames"):
            output_dir = ensure_directory(
                output_cfg.get("dir", "output/macos_frames")
            )

        logger.info(
            "macOS co-sim started — trajectory=%s, renderer=%s",
            sim_cfg.get("trajectory"),
            renderer_cfg.get("backend"),
        )

        frame = 0
        try:
            for pose in simulator.poses():
                image = renderer.render(pose)

                if ros is not None:
                    ros.publish_image(
                        "front_wide_image",
                        image,
                        pose.stamp_sec,
                        pose.stamp_nanosec,
                    )
                    ros.publish_clock(pose.stamp_sec, pose.stamp_nanosec)
                    ros.spin_once(timeout_sec=0.0)

                if output_dir is not None:
                    self._save_frame(output_dir, frame, image)

                if preview is not None:
                    if not self._show_frame(preview, image):
                        logger.info("Preview window closed.")
                        break

                frame += 1
                if frame % 50 == 0:
                    logger.info(
                        "Frame %d — pos=(%.1f, %.1f, %.1f)",
                        frame,
                        *pose.position,
                    )

                if sim_cfg.get("realtime", True):
                    time.sleep(1.0 / float(sim_cfg.get("tick_hz", 20.0)))

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            if preview is not None:
                self._close_preview(preview)
            if ros is not None:
                ros.shutdown()

        logger.info("Finished %d frames.", frame)
        return 0

    def _open_preview(self):
        try:
            import cv2  # noqa: F401

            return "cv2"
        except ImportError:
            logger.info(
                "OpenCV not installed — no live preview. "
                "Install with: pip install opencv-python\n"
                "Or use --save-frames to write images to disk."
            )
            return None

    def _show_frame(self, backend: str, image) -> bool:
        import cv2

        rgb = image[:, :, ::-1]
        cv2.imshow("carla-3dgs-ros (macOS co-sim)", rgb)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def _close_preview(self, backend: str) -> None:
        import cv2

        cv2.destroyAllWindows()

    @staticmethod
    def _save_frame(output_dir: Path, frame: int, image) -> None:
        try:
            import cv2

            path = output_dir / f"frame_{frame:06d}.png"
            cv2.imwrite(str(path), image)
        except ImportError:
            from PIL import Image

            path = output_dir / f"frame_{frame:06d}.png"
            rgb = image[:, :, ::-1]
            Image.fromarray(rgb).save(path)


def run_from_cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="macOS co-simulation: synthetic poses + 3DGS preview (no CARLA).",
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "config"
            / "cosim_macos.yaml"
        ),
    )
    parser.add_argument("--ros", action="store_true", help="Enable ROS2 publishing.")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--export-pose-template",
        metavar="PATH",
        help="Write a sample pose-replay JSON and exit.",
    )
    args = parser.parse_args(argv)

    if args.export_pose_template:
        export_poses_template(args.export_pose_template)
        print(f"Wrote pose template to {args.export_pose_template}")
        return 0

    orchestrator = MacCoSimOrchestrator(
        MacCoSimOptions(
            config_path=args.config,
            enable_ros=args.ros,
            show_preview=not args.no_preview,
            save_frames=args.save_frames,
        )
    )
    return orchestrator.run()