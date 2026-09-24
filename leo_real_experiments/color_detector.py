"""Shared RGB-D color visibility and reach detection for real Leo missions."""

import math
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String


COLORS = ("red", "green", "blue")


class ColorDetector(Node):
    """Publish visibility, reached state, and distance for each color."""

    def __init__(self):
        super().__init__("color_detector")

        self.declare_parameter("rgb_topic", "camera/color/image_raw")
        self.declare_parameter(
            "depth_topic", "camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("reached_distance", 1.0)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("depth_min_m", 0.10)
        self.declare_parameter("depth_max_m", 6.0)
        self.declare_parameter("depth_min_samples", 20)
        self.declare_parameter("depth_max_age_s", 0.5)
        self.declare_parameter("min_color_pixels", 4)
        self.declare_parameter("visible_width_ratio", 0.90)
        self.declare_parameter("visibility_on_frames", 2)
        self.declare_parameter("visibility_off_frames", 5)
        self.declare_parameter("exclusive_dominant_color", True)
        self.declare_parameter("color_min_area_ratio", 0.002)
        self.declare_parameter("red_aruco_marker_id", 0)
        self.declare_parameter("green_aruco_marker_id", 1)
        self.declare_parameter("blue_aruco_marker_id", 2)
        self.declare_parameter("aruco_marker_size_m", 0.10)
        self.declare_parameter("heartbeat_period_s", 5.0)
        self.declare_parameter("stale_frame_threshold_s", 2.0)
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("target_half_extent", 0.075)
        self.declare_parameter("red_target_pose", [-3.0, -1.5, 0.37])
        self.declare_parameter("green_target_pose", [-1.8, 2.6, 1.12])
        self.declare_parameter("blue_target_pose", [2.4, -2.7, 2.31])
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
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_depth_at: Optional[float] = None
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_100
        )
        if hasattr(cv2.aruco, "DetectorParameters"):
            aruco_detector_parameters = cv2.aruco.DetectorParameters()
        else:
            aruco_detector_parameters = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(
                aruco_dictionary, aruco_detector_parameters
            )
        else:
            self.aruco_detector = None
        self.aruco_dictionary = aruco_dictionary
        self.aruco_detector_parameters = aruco_detector_parameters
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
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self.odom_callback,
            10,
        )

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        rgb_topic = str(self.get_parameter("rgb_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        self.create_subscription(Image, rgb_topic, self.rgb_callback, sensor_qos)
        self.create_subscription(
            Image, depth_topic, self.depth_callback, sensor_qos
        )
        # RGB heartbeat logging is intentionally disabled to keep mission
        # supervisor logs focused on events and task progress.
        # heartbeat_period = max(
        #     0.1, float(self.get_parameter("heartbeat_period_s").value)
        # )
        # self.heartbeat_timer = self.create_timer(
        #     heartbeat_period, self.rgb_heartbeat_callback
        # )
        self.get_logger().info(
            f"Color detector listening on RGB '{rgb_topic}' and aligned depth "
            f"'{depth_topic}'; targets use DICT_5X5_100 IDs "
            f"red={self.get_parameter('red_aruco_marker_id').value}, "
            f"green={self.get_parameter('green_aruco_marker_id').value}, "
            f"blue={self.get_parameter('blue_aruco_marker_id').value}, "
            f"marker size={self.get_parameter('aruco_marker_size_m').value} m; "
            "reached uses median target-mask depth"
        )

    def odom_callback(self, message: Odometry):
        if self.task_completed:
            return
        self.robot_xy = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )

    def target_surface_distance(self, color: str) -> float:
        """Return planar distance from the robot point to a rotated target box."""
        if self.robot_xy is None:
            return float("nan")
        target = list(self.get_parameter(f"{color}_target_pose").value)
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

    def depth_callback(self, message: Image):
        """Store the newest aligned depth frame in metres."""
        encoding = message.encoding.lower()
        if encoding in ("16uc1", "mono16"):
            row_values = message.step // np.dtype(np.uint16).itemsize
            count = row_values * message.height
            depth = np.frombuffer(message.data, dtype=np.uint16, count=count)
            depth = depth.reshape(message.height, row_values)[:, : message.width]
            depth = depth.astype(np.float32) * float(
                self.get_parameter("depth_scale").value
            )
        elif encoding == "32fc1":
            row_values = message.step // np.dtype(np.float32).itemsize
            count = row_values * message.height
            depth = np.frombuffer(message.data, dtype=np.float32, count=count)
            depth = depth.reshape(message.height, row_values)[:, : message.width].copy()
        else:
            self.get_logger().warning(
                f"Unsupported depth encoding '{message.encoding}'",
                throttle_duration_sec=2.0,
            )
            return
        self.latest_depth_m = depth
        self.latest_depth_at = time.monotonic()

    def color_depth_distance(self, color_mask: np.ndarray) -> float:
        """Return median valid aligned depth under a color mask."""
        if self.latest_depth_m is None or self.latest_depth_at is None:
            return float("nan")
        max_age = float(self.get_parameter("depth_max_age_s").value)
        if time.monotonic() - self.latest_depth_at > max_age:
            return float("nan")

        depth = self.latest_depth_m
        if color_mask.shape != depth.shape:
            y_indices = np.linspace(
                0, color_mask.shape[0] - 1, depth.shape[0]
            ).astype(np.intp)
            x_indices = np.linspace(
                0, color_mask.shape[1] - 1, depth.shape[1]
            ).astype(np.intp)
            color_mask = color_mask[np.ix_(y_indices, x_indices)]

        minimum = float(self.get_parameter("depth_min_m").value)
        maximum = float(self.get_parameter("depth_max_m").value)
        valid = color_mask & np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
        samples = depth[valid]
        required = max(1, int(self.get_parameter("depth_min_samples").value))
        if samples.size < required:
            return float("nan")
        return float(np.median(samples))

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
        """Return filled masks for the three configured ArUco target IDs."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dictionary,
                parameters=self.aruco_detector_parameters,
            )

        masks = {
            color: np.zeros(gray.shape, dtype=np.uint8) for color in COLORS
        }
        if ids is None:
            return {color: mask.astype(bool) for color, mask in masks.items()}

        colors_by_id = {
            int(self.get_parameter(f"{color}_aruco_marker_id").value): color
            for color in COLORS
        }
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            color = colors_by_id.get(int(marker_id))
            if color is None:
                continue
            polygon = np.rint(marker_corners.reshape(-1, 2)).astype(np.int32)
            cv2.fillConvexPoly(masks[color], polygon, 1)
        return {color: mask.astype(bool) for color, mask in masks.items()}

    def visible_region(self, mask: np.ndarray) -> np.ndarray:
        ratio = float(self.get_parameter("visible_width_ratio").value)
        width = max(1, int(round(mask.shape[1] * np.clip(ratio, 0.0, 1.0))))
        left = (mask.shape[1] - width) // 2
        return mask[:, left : left + width]

    def publish_color(self, color: str, visible: bool, distance: float):
        if self.task_completed:
            return
        reached_distance = float(self.get_parameter("reached_distance").value)
        reached = visible and np.isfinite(distance) and distance <= reached_distance
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
        """Detect colors and measure their aligned-depth distance."""
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
        min_area_ratio = float(
            self.get_parameter("color_min_area_ratio").value
        )
        pixel_counts = {
            color: int(np.count_nonzero(masks[color])) for color in COLORS
        }
        required_pixels = max(
            min_pixels,
            int(np.ceil(min_area_ratio * next(iter(masks.values())).size)),
        )
        candidates = [
            color for color in COLORS if pixel_counts[color] >= required_pixels
        ]
        dominant_color = (
            max(candidates, key=pixel_counts.get) if candidates else None
        )
        exclusive = bool(
            self.get_parameter("exclusive_dominant_color").value
        )
        for color in COLORS:
            detected = pixel_counts[color] >= required_pixels
            if exclusive:
                detected = detected and color == dominant_color
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
            visible_by_color[color] = self.visibility_state[color] and (
                not exclusive or color == dominant_color
            )

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
            distance = self.color_depth_distance(full_masks[color])
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
