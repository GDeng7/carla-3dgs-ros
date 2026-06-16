"""ROS2/DDS bridge — publish NuRec/CARLA sensor data to ROS topics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RosBridgeConfig:
    namespace: str = "carla"
    ego_name: str = "ego"
    frame_id: str = "map"
    publish_rate_hz: int = 20
    topics: dict[str, str] = field(default_factory=dict)


class RosBridge:
    """
    Publish camera images and vehicle state over ROS2 (DDS underneath).

    When CARLA is launched with --ros2, CARLA sensors with enable_for_ros()
    publish directly on DDS. This bridge adds a Python-side publisher for
    NuRec-rendered frames and supplemental state topics.
    """

    def __init__(self, config: RosBridgeConfig) -> None:
        self.config = config
        self._rclpy: Any = None
        self._node: Any = None
        self._publishers: dict[str, Any] = {}
        self._clock_pub: Any = None
        self._initialized = False

    @staticmethod
    def available() -> bool:
        try:
            import rclpy  # noqa: F401

            return True
        except ImportError:
            return False

    def _topic(self, suffix: str) -> str:
        return f"/{self.config.namespace}/{self.config.ego_name}/{suffix}"

    def initialize(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Header

        self._rclpy = rclpy
        rclpy.init(args=None)
        self._node = Node("carla_3dgs_ros_bridge")

        topic_map = {
            "front_wide_image": Image,
            "front_tele_image": Image,
            "cross_left_image": Image,
            "cross_right_image": Image,
            "rear_left_image": Image,
            "rear_right_image": Image,
            "odometry": Odometry,
        }

        for key, msg_type in topic_map.items():
            suffix = self.config.topics.get(key, key)
            full_topic = self._topic(suffix)
            self._publishers[key] = self._node.create_publisher(msg_type, full_topic, 10)
            logger.info("ROS2 publisher: %s", full_topic)

        clock_suffix = self.config.topics.get("clock", "clock")
        self._clock_pub = self._node.create_publisher(Clock, self._topic(clock_suffix), 10)
        self._initialized = True
        self._Header = Header

    def publish_image(
        self,
        camera_key: str,
        image_bgr: np.ndarray,
        stamp_sec: int,
        stamp_nanosec: int,
    ) -> None:
        if not self._initialized:
            return

        from sensor_msgs.msg import Image

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape {image_bgr.shape}")

        height, width = image_bgr.shape[:2]
        msg = Image()
        msg.header = self._Header()
        msg.header.stamp.sec = stamp_sec
        msg.header.stamp.nanosec = stamp_nanosec
        msg.header.frame_id = f"{self.config.ego_name}/{camera_key}"
        msg.height = height
        msg.width = width
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = width * 3
        msg.data = image_bgr.tobytes()

        publisher = self._publishers.get(camera_key)
        if publisher is not None:
            publisher.publish(msg)

    def publish_odometry(
        self,
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        stamp_sec: int,
        stamp_nanosec: int,
    ) -> None:
        if not self._initialized:
            return

        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Quaternion
        import math

        msg = Odometry()
        msg.header = self._Header()
        msg.header.stamp.sec = stamp_sec
        msg.header.stamp.nanosec = stamp_nanosec
        msg.header.frame_id = self.config.frame_id
        msg.child_frame_id = f"{self.config.ego_name}/base_link"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        half_yaw = yaw_rad / 2.0
        msg.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(half_yaw),
            w=math.cos(half_yaw),
        )
        self._publishers["odometry"].publish(msg)

    def publish_clock(self, stamp_sec: int, stamp_nanosec: int) -> None:
        if not self._initialized or self._clock_pub is None:
            return
        from rosgraph_msgs.msg import Clock
        from builtin_interfaces.msg import Time

        msg = Clock()
        msg.clock = Time(sec=stamp_sec, nanosec=stamp_nanosec)
        self._clock_pub.publish(msg)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        if self._initialized and self._rclpy is not None:
            self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        self._initialized = False
        logger.info("ROS2 bridge shut down.")