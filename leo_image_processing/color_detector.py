"""Shared color detection and world-space reach detection for Leo missions."""

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String
from tf2_msgs.msg import TFMessage


COLORS = ("red", "green", "blue")


class ColorDetector(Node):
    """Publish visibility, reached state, and distance for each color."""

    def __init__(self):
        super().__init__("color_detector")

        self.declare_parameter("rgb_topic", "depth_camera/image")
        self.declare_parameter("reached_distance", 1.0)
        self.declare_parameter("min_color_pixels", 4)
        self.declare_parameter("visible_width_ratio", 0.90)
        self.declare_parameter("visibility_on_frames", 2)
        self.declare_parameter("visibility_off_frames", 5)
        self.declare_parameter("min_channel", 80)
        self.declare_parameter("channel_margin", 35)
        self.declare_parameter("heartbeat_period_s", 5.0)
        self.declare_parameter("stale_frame_threshold_s", 2.0)
        self.declare_parameter("pose_topic", "/world/random_world/dynamic_pose/info")
        self.declare_parameter("target_half_extent", 0.075)
        self.declare_parameter("red_target_pose", [-2.0, -1.5, 0.37])
        self.declare_parameter("green_target_pose", [-1.8, 2.6, 1.12])
        self.declare_parameter("blue_target_pose", [2.4, -2.7, 2.31])
        # Optional flattened [x, y, yaw, ...] list for same-color targets.
        # Empty preserves the original single-target behavior.
        for color in COLORS:
            self.declare_parameter(
                f"{color}_target_poses",
                [],
                ParameterDescriptor(dynamic_typing=True),
            )
        default_robot_name = self.get_namespace().strip("/").split("/")[-1]
        self.declare_parameter("robot_name", default_robot_name)

        self.robot_xy: Optional[Tuple[float, float]] = None
        self.robot_name = str(self.get_parameter("robot_name").value).strip("/")
        self.task_completed = False
        self.rgb_frame_count = 0
        self.last_rgb_frame_at: Optional[float] = None
        self.last_heartbeat_at = time.monotonic()
        self.last_heartbeat_frame_count = 0
        self.visibility_state = {color: False for color in COLORS}
        self.visible_frame_counts = {color: 0 for color in COLORS}
        self.missing_frame_counts = {color: 0 for color in COLORS}
        self.last_horizontal_offsets = {
            color: float("nan") for color in COLORS
        }
        self.visible_publishers = {
            color: self.create_publisher(Bool, f"{color}_visible", 10)
            for color in COLORS
        }
        self.reached_publishers = {
            color: self.create_publisher(Bool, f"{color}_reached", 10)
            for color in COLORS
        }
        self.distance_publishers = {
            color: self.create_publisher(Float32, f"{color}_distance", 10)
            for color in COLORS
        }
        self.horizontal_offset_publishers = {
            color: self.create_publisher(
                Float32, f"{color}_horizontal_offset", 10
            )
            for color in COLORS
        }
        self.event_publisher = self.create_publisher(String, "color_events", 10)
        self.any_color_visible_publisher = self.create_publisher(
            Bool, "color_visible", 10
        )
        self.create_subscription(
            Bool,
            f"/{self.robot_name}/task_complete",
            self.task_complete_callback,
            10,
        )
        self.create_subscription(
            TFMessage,
            str(self.get_parameter("pose_topic").value),
            self.pose_callback,
            10,
        )

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.create_subscription(Image, rgb_topic, self.rgb_callback, sensor_qos)
        heartbeat_period = max(
            0.1, float(self.get_parameter("heartbeat_period_s").value)
        )
        self.heartbeat_timer = self.create_timer(
            heartbeat_period, self.rgb_heartbeat_callback
        )
        self.get_logger().info(
            f"Color detector listening on RGB '{rgb_topic}'; reached uses world poses"
        )

    def pose_callback(self, message: TFMessage):
        if self.task_completed:
            return
        for transform in message.transforms:
            frame_name = transform.child_frame_id.strip("/")
            if frame_name == self.robot_name:
                self.robot_xy = (
                    float(transform.transform.translation.x),
                    float(transform.transform.translation.y),
                )
                return

    def target_surface_distance(self, color: str) -> float:
        """Return planar distance from the robot point to a rotated target box."""
        if self.robot_xy is None:
            return float("nan")
        flattened = list(
            self.get_parameter(f"{color}_target_poses").value
        )
        targets = (
            [flattened[index:index + 3] for index in range(0, len(flattened), 3)]
            if flattened and len(flattened) % 3 == 0
            else [list(self.get_parameter(f"{color}_target_pose").value)]
        )
        return min(self._target_surface_distance(target) for target in targets)

    def _target_surface_distance(self, target) -> float:
        target_x, target_y, yaw = (float(value) for value in target)
        dx = self.robot_xy[0] - target_x
        dy = self.robot_xy[1] - target_y
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        half_extent = float(self.get_parameter("target_half_extent").value)
        outside_x = max(abs(local_x) - half_extent, 0.0)
        outside_y = max(abs(local_y) - half_extent, 0.0)
        return math.hypot(outside_x, outside_y)

    @staticmethod
    def rgb_from_message(message: Image) -> Optional[np.ndarray]:
        enc = message.encoding.lower()
        channels = 4 if enc in ("rgba8", "bgra8") else 3
        if enc not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            return None
        row_values = message.step
        count = row_values * message.height
        image = np.frombuffer(message.data, dtype=np.uint8, count=count)
        image = image.reshape(message.height, row_values)
        image = image[:, : message.width * channels].reshape(
            message.height, message.width, channels
        )
        if enc in ("bgr8", "bgra8"):
            image = image[:, :, [2, 1, 0]]
        else:
            image = image[:, :, :3]
        return image

    def color_masks(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        values = rgb.astype(np.int16, copy=False)
        red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
        minimum = int(self.get_parameter("min_channel").value)
        margin = int(self.get_parameter("channel_margin").value)
        return {
            "red": (red >= minimum) & (red >= green + margin) & (red >= blue + margin),
            "green": (green >= minimum)
            & (green >= red + margin)
            & (green >= blue + margin),
            "blue": (blue >= minimum)
            & (blue >= red + margin)
            & (blue >= green + margin),
        }

    def visible_region(self, mask: np.ndarray) -> np.ndarray:
        ratio = float(self.get_parameter("visible_width_ratio").value)
        width = max(1, int(round(mask.shape[1] * np.clip(ratio, 0.0, 1.0))))
        left = (mask.shape[1] - width) // 2
        return mask[:, left : left + width]

    def publish_color(self, color: str, visible: bool, distance: float):
        if self.task_completed:
            return
        reached_distance = float(self.get_parameter("reached_distance").value)
        reached = np.isfinite(distance) and distance <= reached_distance
        self.visible_publishers[color].publish(Bool(data=visible))
        self.reached_publishers[color].publish(Bool(data=bool(reached)))
        self.distance_publishers[color].publish(Float32(data=distance))
        if reached:
            event = f"EV_{color}_reached"
        elif visible:
            event = f"EV_{color}_visible"
        else:
            event = f"EV_{color}_not_visible"
        self.event_publisher.publish(String(data=event))

    def rgb_callback(self, message: Image):
        if self.task_completed:
            return
        self.rgb_frame_count += 1
        self.last_rgb_frame_at = time.monotonic()
        rgb = self.rgb_from_message(message)
        if rgb is None:
            self.get_logger().warning(
                f"Unsupported RGB encoding '{message.encoding}'",
                throttle_duration_sec=2.0,
            )
            return
        self._process_rgb_frame(rgb)

    def rgb_heartbeat_callback(self):
        """Report whether RGB callbacks are still being received."""
        if self.task_completed:
            return

        now = time.monotonic()
        interval = max(now - self.last_heartbeat_at, 1e-6)
        frames = self.rgb_frame_count - self.last_heartbeat_frame_count
        rate_hz = frames / interval
        age_s = (
            float("inf")
            if self.last_rgb_frame_at is None
            else max(0.0, now - self.last_rgb_frame_at)
        )
        message = (
            "RGB HEARTBEAT "
            f"frames_total={self.rgb_frame_count} "
            f"recent_rate_hz={rate_hz:.3f} "
            f"last_frame_age_s={age_s:.3f}"
        )
        stale_threshold = float(
            self.get_parameter("stale_frame_threshold_s").value
        )
        if age_s > stale_threshold:
            self.get_logger().warning(message + " status=STALE")
        else:
            self.get_logger().info(message + " status=OK")

        self.last_heartbeat_at = now
        self.last_heartbeat_frame_count = self.rgb_frame_count

    def _process_rgb_frame(self, rgb: np.ndarray):
        """Process color independently of the depth camera."""
        if self.task_completed:
            return
        min_pixels = int(self.get_parameter("min_color_pixels").value)
        full_masks = self.color_masks(rgb)
        masks = {
            color: self.visible_region(mask)
            for color, mask in full_masks.items()
        }
        on_frames = max(
            1, int(self.get_parameter("visibility_on_frames").value)
        )
        off_frames = max(
            1, int(self.get_parameter("visibility_off_frames").value)
        )
        visible_by_color = {}
        for color in COLORS:
            detected = int(np.count_nonzero(masks[color])) >= min_pixels
            if detected:
                self.visible_frame_counts[color] += 1
                self.missing_frame_counts[color] = 0
                if self.visible_frame_counts[color] >= on_frames:
                    self.visibility_state[color] = True
            else:
                self.visible_frame_counts[color] = 0
                self.missing_frame_counts[color] += 1
                if self.missing_frame_counts[color] >= off_frames:
                    self.visibility_state[color] = False
            visible_by_color[color] = self.visibility_state[color]

            pixel_columns = np.flatnonzero(full_masks[color]) % rgb.shape[1]
            if pixel_columns.size >= min_pixels:
                horizontal_offset = float(
                    np.clip(
                        (2.0 * float(np.mean(pixel_columns))
                         / max(1, rgb.shape[1] - 1)) - 1.0,
                        -1.0,
                        1.0,
                    )
                )
                self.last_horizontal_offsets[color] = horizontal_offset
            elif visible_by_color[color]:
                horizontal_offset = self.last_horizontal_offsets[color]
            else:
                horizontal_offset = float("nan")
            self.horizontal_offset_publishers[color].publish(
                Float32(data=horizontal_offset)
            )
        self.any_color_visible_publisher.publish(
            Bool(data=any(visible_by_color.values()))
        )
        for color in COLORS:
            visible = visible_by_color[color]
            distance = self.target_surface_distance(color)
            self.publish_color(color, visible, distance)

    def task_complete_callback(self, message: Bool):
        if not message.data:
            return
        self.task_completed = True
        self.get_logger().info("Task complete; color publications stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
