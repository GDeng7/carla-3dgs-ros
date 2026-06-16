"""NuRec gRPC Docker container lifecycle management."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NuRecContainerConfig:
    image: str
    grpc_port: int = 46435
    cuda_visible_devices: str = "0"
    container_name: str = "carla-nurec-grpc"


class NuRecContainer:
    """Manage the NuRec neural rendering Docker container."""

    def __init__(self, config: NuRecContainerConfig) -> None:
        self.config = config
        self._running = False

    @staticmethod
    def prerequisites_ok() -> tuple[bool, str]:
        if shutil.which("docker") is None:
            return False, "Docker is not installed or not on PATH."
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            return False, f"Docker daemon not reachable: {exc.stderr.strip()}"
        return True, "ok"

    def start(self) -> None:
        ok, message = self.prerequisites_ok()
        if not ok:
            raise RuntimeError(message)

        self.stop()

        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--gpus",
            "all",
            "--name",
            self.config.container_name,
            "-p",
            f"{self.config.grpc_port}:{self.config.grpc_port}",
            "-e",
            f"CUDA_VISIBLE_DEVICES={self.config.cuda_visible_devices}",
            self.config.image,
        ]
        logger.info("Starting NuRec container: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        self._running = True
        self._wait_until_ready(timeout=120)

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={self.config.container_name}",
                    "--format",
                    "{{.Status}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            status = result.stdout.strip()
            if status.startswith("Up"):
                logger.info("NuRec container is running (%s)", status)
                return
            time.sleep(2.0)
        raise TimeoutError("NuRec container did not become ready in time.")

    def stop(self) -> None:
        subprocess.run(
            ["docker", "rm", "-f", self.config.container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        self._running = False

    @property
    def running(self) -> bool:
        return self._running