"""Record the fixed Gazebo overhead camera into a simulation run artifact."""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class SimulationVideoRecorder(Node):
    def __init__(self) -> None:
        super().__init__("simulation_video_recorder")
        results_root = Path(str(self.declare_parameter("results_dir", "").value))
        run_id = str(self.declare_parameter("run_id", "run_unknown").value)
        yaml_path = Path(str(self.declare_parameter("metadata_yaml_path", "").value))
        robot_count = str(self.declare_parameter("total_robots", 0).value)
        topic = str(
            self.declare_parameter("image_topic", "/top_view_camera/image").value
        )
        self.fps = float(self.declare_parameter("fps", 15.0).value)
        self.enabled = bool(self.declare_parameter("enabled", False).value)
        self.writer: cv2.VideoWriter | None = None
        filename = (
            "baseline_simulation.mp4"
            if "baseline" in results_root.parts
            else "simulation.mp4"
        )
        self.output_path = (
            results_root
            / (yaml_path.stem or "unknown_yaml")
            / f"robots_{robot_count}"
            / run_id
            / filename
        )
        if not self.enabled:
            self.get_logger().info("Simulation video recording disabled for this run")
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.subscription = self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f"Recording simulation video to {self.output_path}")

    @staticmethod
    def _frame(message: Image) -> np.ndarray:
        channels = {
            "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1
        }.get(message.encoding.lower())
        if channels is None:
            raise ValueError(f"Unsupported image encoding: {message.encoding}")
        row_bytes = message.width * channels
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step
        )[:, :row_bytes]
        frame = rows.reshape(message.height, message.width, channels)
        encoding = message.encoding.lower()
        if encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if encoding == "mono8":
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def _on_image(self, message: Image) -> None:
        try:
            frame = self._frame(message)
            if self.writer is None:
                height, width = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.fps,
                    (width, height),
                )
                if not self.writer.isOpened():
                    raise RuntimeError(f"Could not open video output {self.output_path}")
            self.writer.write(frame)
        except Exception as error:
            self.get_logger().error(f"Could not record camera frame: {error}")

    def destroy_node(self) -> bool:
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"Saved simulation video: {self.output_path}")
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = SimulationVideoRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
