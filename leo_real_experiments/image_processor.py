"""ROS 2 depth-image zone detector based on the closest pixel only."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String


class ImageProcessor(Node):
    """Publish the obstacle zone containing the closest valid depth pixel."""

    def __init__(self):
        super().__init__("image_processor")

        self.declare_parameter("depth_topic", "camera/depth/image_rect_raw")
        self.declare_parameter("obstacle_threshold", 0.50)
        self.declare_parameter("min_depth", 0.08)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("crop_y0_frac", 0.30)
        self.declare_parameter("crop_y1_frac", 0.80)

        depth_topic = str(self.get_parameter("depth_topic").value)
        robot_name = self.get_namespace().strip("/").split("/")[-1]
        self.task_completed = False
        self.zone_publisher = self.create_publisher(String, "detected_zones", 10)
        self.wall_publisher = self.create_publisher(Float32, "parking_wall_distance", 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.create_subscription(
            Bool,
            f"/{robot_name}/task_complete",
            self.task_complete_callback,
            10,
        )

        self.get_logger().info(f"Argmin zone detector listening on '{depth_topic}'")

    def publish_zone(self, zone: str):
        if self.task_completed:
            return
        self.zone_publisher.publish(String(data=zone))

    def task_complete_callback(self, message: Bool):
        if not message.data:
            return
        self.task_completed = True
        self.get_logger().info("Task complete; obstacle publications stopped.")

    @staticmethod
    def image_to_depth(message: Image):
        """Return a metre-valued depth array for supported ROS encodings."""
        encoding = message.encoding.upper()
        if encoding not in {"32FC1", "16UC1"}:
            return None
        dtype = np.dtype(np.float32 if encoding == "32FC1" else np.uint16)
        dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
        row_values = message.step // dtype.itemsize
        required_values = row_values * message.height
        depth = np.frombuffer(message.data, dtype=dtype, count=required_values)
        depth = depth.reshape(message.height, row_values)[:, : message.width]
        depth = depth.astype(np.float32, copy=False)
        if encoding == "16UC1":
            depth *= 0.001
        return depth

    def depth_callback(self, message: Image):
        if self.task_completed:
            return
        depth = self.image_to_depth(message)
        if depth is None:
            self.get_logger().warning(
                f"Unsupported depth encoding '{message.encoding}'",
                throttle_duration_sec=2.0,
            )
            return
        if depth.size == 0:
            return

        y0_fraction = float(self.get_parameter("crop_y0_frac").value)
        y1_fraction = float(self.get_parameter("crop_y1_frac").value)
        y0 = int(np.clip(y0_fraction, 0.0, 1.0) * depth.shape[0])
        y1 = int(np.clip(y1_fraction, 0.0, 1.0) * depth.shape[0])
        if y1 <= y0:
            y0, y1 = 0, depth.shape[0]

        cropped = depth[y0:y1]
        self.wall_publisher.publish(Float32(data=self.wall_distance(cropped)))
        min_depth = float(self.get_parameter("min_depth").value)
        max_depth = float(self.get_parameter("max_depth").value)
        valid = np.isfinite(cropped) & (cropped > min_depth) & (cropped < max_depth)

        # Invalid pixels become infinity, so a single argmin selects the
        # closest valid depth pixel in the complete detection region.
        candidates = np.where(valid, cropped, np.inf)
        closest_index = int(np.argmin(candidates))
        closest_depth = float(candidates.flat[closest_index])

        threshold = float(self.get_parameter("obstacle_threshold").value)
        if not np.isfinite(closest_depth) or closest_depth >= threshold:
            zone = "CLEAR"
        else:
            _, x = np.unravel_index(closest_index, candidates.shape)
            one_third = candidates.shape[1] / 3.0
            if x < one_third:
                zone = "LEFT"
            elif x < 2.0 * one_third:
                zone = "CORNER"
            else:
                zone = "RIGHT"

        self.publish_zone(zone)

    @staticmethod
    def wall_distance(depth):
        """Require a broad, approximately even surface across the front view."""
        if depth.size == 0 or depth.shape[1] < 5:
            return float("nan")
        front = depth[:, int(depth.shape[1] * 0.2):int(depth.shape[1] * 0.8)]
        distances = []
        for strip in np.array_split(front, 3, axis=1):
            valid = np.isfinite(strip) & (strip > 0.1) & (strip < 6.0)
            if np.count_nonzero(valid) < 0.7 * strip.size:
                return float("nan")
            distances.append(float(np.median(strip[valid])))
        # Allow perspective variation at longer distances; an oblique wall
        # should not disappear merely because the rover is a few metres away.
        if max(distances) - min(distances) > max(0.35, 0.25 * np.median(distances)):
            return float("nan")
        return min(distances)


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
