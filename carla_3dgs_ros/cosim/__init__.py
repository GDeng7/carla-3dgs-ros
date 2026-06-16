"""Co-simulation: CARLA physics + separate 3DGS renderer + ROS2/DDS."""

from carla_3dgs_ros.cosim.orchestrator import CoSimOrchestrator, CoSimOptions

__all__ = ["CoSimOrchestrator", "CoSimOptions"]