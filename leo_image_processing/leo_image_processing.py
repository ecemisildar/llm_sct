"""ROS 2 depth-image zone detector based on the closest pixel only."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ImageProcessor(Node):
    """Publish the obstacle zone containing the closest valid depth pixel."""

    def __init__(self):
        super().__init__("image_processor")

        self.declare_parameter("depth_topic", "depth_camera/depth_image")
        self.declare_parameter("obstacle_threshold", 0.70)
        self.declare_parameter("min_depth", 0.08)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("crop_y0_frac", 0.30)
        self.declare_parameter("crop_y1_frac", 0.80)

        depth_topic = str(self.get_parameter("depth_topic").value)
        self.zone_publisher = self.create_publisher(
            String, "detected_zones", 10
        )
        self.depth_subscription = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10
        )

        self.get_logger().info(
            f"Argmin zone detector listening on '{depth_topic}'"
        )

    @staticmethod
    def image_to_depth(message: Image):
        """Return a metre-valued depth array for supported ROS encodings."""
        if message.encoding == "32FC1":
            dtype = np.dtype(np.float32)
            scale = 1.0
        else:
            return None

        dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
        row_values = message.step // dtype.itemsize
        required_values = row_values * message.height
        depth = np.frombuffer(message.data, dtype=dtype, count=required_values)
        depth = depth.reshape(message.height, row_values)[:, : message.width]
        return depth.astype(np.float32, copy=False) * scale

    def depth_callback(self, message: Image):
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

        zone_message = String()
        zone_message.data = zone
        self.zone_publisher.publish(zone_message)


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
