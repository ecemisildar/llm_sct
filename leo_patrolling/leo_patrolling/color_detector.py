"""Detect red, green, and blue regions and estimate their depth."""

from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String


COLORS = ("red", "green", "blue")


class ColorDetector(Node):
    """Publish visibility, reached state, and distance for each target color."""

    def __init__(self):
        super().__init__("color_detector")

        self.declare_parameter("rgb_topic", "depth_camera/image")
        self.declare_parameter("depth_topic", "depth_camera/depth_image")
        self.declare_parameter("reached_distance", 0.75)
        self.declare_parameter("min_color_pixels", 150)
        self.declare_parameter("min_channel", 80)
        self.declare_parameter("channel_margin", 35)
        self.declare_parameter("min_depth", 0.08)
        self.declare_parameter("max_depth", 10.0)

        self.latest_depth: Optional[np.ndarray] = None
        self.last_color_event = {color: None for color in COLORS}
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
        self.event_publisher = self.create_publisher(String, "color_events", 10)
        self.any_color_visible_publisher = self.create_publisher(
            Bool, "color_visible", 10
        )

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        rgb_topic = str(self.get_parameter("rgb_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        self.create_subscription(Image, depth_topic, self.depth_callback, sensor_qos)
        self.create_subscription(Image, rgb_topic, self.rgb_callback, sensor_qos)
        self.get_logger().info(
            f"Color detector listening on RGB '{rgb_topic}' and depth '{depth_topic}'"
        )

    @staticmethod
    def depth_from_message(message: Image) -> Optional[np.ndarray]:
        if message.encoding != "32FC1":
            return None
        dtype = np.dtype(np.float32).newbyteorder(
            ">" if message.is_bigendian else "<"
        )
        row_values = message.step // dtype.itemsize
        count = row_values * message.height
        image = np.frombuffer(message.data, dtype=dtype, count=count)
        return image.reshape(message.height, row_values)[:, : message.width]

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

    def depth_callback(self, message: Image):
        depth = self.depth_from_message(message)
        if depth is None:
            self.get_logger().warning(
                f"Unsupported depth encoding '{message.encoding}'",
                throttle_duration_sec=2.0,
            )
            return
        self.latest_depth = depth

    def color_masks(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        values = rgb.astype(np.int16, copy=False)
        red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
        minimum = int(self.get_parameter("min_channel").value)
        margin = int(self.get_parameter("channel_margin").value)
        return {
            "red": (red >= minimum) & (red >= green + margin) & (red >= blue + margin),
            "green": (green >= minimum) & (green >= red + margin) & (green >= blue + margin),
            "blue": (blue >= minimum) & (blue >= red + margin) & (blue >= green + margin),
        }

    def distance_for_mask(self, mask: np.ndarray) -> float:
        depth = self.latest_depth
        if depth is None or depth.shape != mask.shape:
            return float("nan")
        min_depth = float(self.get_parameter("min_depth").value)
        max_depth = float(self.get_parameter("max_depth").value)
        values = depth[mask]
        valid = values[np.isfinite(values) & (values > min_depth) & (values < max_depth)]
        if valid.size == 0:
            return float("nan")
        # A low percentile favors the front face while rejecting isolated noise.
        return float(np.percentile(valid, 20.0))

    def publish_color(self, color: str, visible: bool, distance: float):
        reached_distance = float(self.get_parameter("reached_distance").value)
        reached = visible and np.isfinite(distance) and distance <= reached_distance
        self.visible_publishers[color].publish(Bool(data=visible))
        self.reached_publishers[color].publish(Bool(data=bool(reached)))
        self.distance_publishers[color].publish(Float32(data=distance))
        event = None
        if reached:
            event = f"EV_{color}_reached"
        elif visible:
            event = f"EV_{color}_visible"
        if event != self.last_color_event[color]:
            if event is not None:
                self.event_publisher.publish(String(data=event))
            self.last_color_event[color] = event

    def rgb_callback(self, message: Image):
        rgb = self.rgb_from_message(message)
        if rgb is None:
            self.get_logger().warning(
                f"Unsupported RGB encoding '{message.encoding}'",
                throttle_duration_sec=2.0,
            )
            return
        min_pixels = int(self.get_parameter("min_color_pixels").value)
        masks = self.color_masks(rgb)
        visible_by_color = {
            color: int(np.count_nonzero(mask)) >= min_pixels
            for color, mask in masks.items()
        }
        self.any_color_visible_publisher.publish(
            Bool(data=any(visible_by_color.values()))
        )
        for color, mask in masks.items():
            visible = visible_by_color[color]
            distance = self.distance_for_mask(mask) if visible else float("nan")
            self.publish_color(color, visible, distance)


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
