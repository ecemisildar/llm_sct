"""Distinguish colored pickup boxes from delivery-zone sticks in RGB images."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import rclpy
from std_msgs.msg import Bool, Float32, String

from color_detector import COLORS, ColorDetector


KINDS = ("object", "zone")


class DeliveryShapeDetector(ColorDetector):
    """Publish delivery object/zone events based on colored component shape."""

    def __init__(self):
        super().__init__()
        self.declare_parameter("stick_min_aspect_ratio", 1.5)
        self.declare_parameter("min_shape_pixels", 6)
        # Compact boxes are pickup objects; tall sticks are delivery zones.
        self.declare_parameter("red_object_pose", [-3.5, 3.7, 0.0])
        self.declare_parameter("green_object_pose", [0.0, -3.7, 0.0])
        self.declare_parameter("blue_object_pose", [3.5, 3.7, 0.0])
        self.declare_parameter("red_zone_pose", [-3.0, -1.5, 0.0])
        self.declare_parameter("green_zone_pose", [-1.8, 2.6, 0.0])
        self.declare_parameter("blue_zone_pose", [2.4, -2.7, 0.0])
        self.declare_parameter("object_half_extent", 0.125)
        # Half-width of the complete pickup footprint along the world X axis.
        # It equals object_half_extent for the original single-box task and is
        # widened by the complex-task launch for its three-box row.
        self.declare_parameter("object_cluster_half_width", 0.125)
        self.declare_parameter("zone_half_extent", 0.125)
        self.shape_offset_publishers = {
            (color, kind): self.create_publisher(
                Float32, f"{color}_{kind}_horizontal_offset", 10
            )
            for color in COLORS
            for kind in KINDS
        }
        self.shape_visibility = {
            (color, kind): False for color in COLORS for kind in KINDS
        }
        self.shape_on_counts = {
            (color, kind): 0 for color in COLORS for kind in KINDS
        }
        self.shape_off_counts = {
            (color, kind): 0 for color in COLORS for kind in KINDS
        }
        self.get_logger().info(
            "Delivery shape classification enabled: compact=object box, "
            "tall/narrow=delivery-zone stick."
        )

    @staticmethod
    def connected_components(mask: np.ndarray, minimum_pixels: int):
        """Return bounding boxes for 8-connected true-pixel components."""
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        for start_y, start_x in np.argwhere(mask):
            if visited[start_y, start_x]:
                continue
            queue = deque([(int(start_y), int(start_x))])
            visited[start_y, start_x] = True
            min_x = max_x = int(start_x)
            min_y = max_y = int(start_y)
            count = 0
            sum_x = 0
            while queue:
                y, x = queue.popleft()
                count += 1
                sum_x += x
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < height
                            and 0 <= nx < width
                            and mask[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            if count >= minimum_pixels:
                components.append(
                    {
                        "pixels": count,
                        "width": max_x - min_x + 1,
                        "height": max_y - min_y + 1,
                        "center_x": sum_x / count,
                    }
                )
        return components

    def target_distance(self, color: str, kind: str) -> float:
        if self.robot_xy is None:
            return float("nan")
        parameter = f"{color}_{kind}_pose"
        target = list(self.get_parameter(parameter).value)
        dx = self.robot_xy[0] - float(target[0])
        dy = self.robot_xy[1] - float(target[1])
        if kind == "object":
            half_x = float(
                self.get_parameter("object_cluster_half_width").value
            )
            half_y = float(self.get_parameter("object_half_extent").value)
        else:
            half_x = half_y = float(
                self.get_parameter("zone_half_extent").value
            )
        # Distance to the nearest point of the axis-aligned target footprint.
        # Unlike subtracting one radius from the center distance, this covers
        # every box in the complex task's horizontal three-box cluster.
        outside_x = max(abs(dx) - half_x, 0.0)
        outside_y = max(abs(dy) - half_y, 0.0)
        return math.hypot(outside_x, outside_y)

    def _update_shape_visibility(
        self, color: str, kind: str, detected: bool
    ) -> bool:
        key = (color, kind)
        on_frames = max(
            1, int(self.get_parameter("visibility_on_frames").value)
        )
        off_frames = max(
            1, int(self.get_parameter("visibility_off_frames").value)
        )
        if detected:
            self.shape_on_counts[key] += 1
            self.shape_off_counts[key] = 0
            if self.shape_on_counts[key] >= on_frames:
                self.shape_visibility[key] = True
        else:
            self.shape_on_counts[key] = 0
            self.shape_off_counts[key] += 1
            if self.shape_off_counts[key] >= off_frames:
                self.shape_visibility[key] = False
        return self.shape_visibility[key]

    def _process_rgb_frame(self, rgb: np.ndarray):
        if self.task_completed:
            return
        minimum_pixels = max(
            1, int(self.get_parameter("min_shape_pixels").value)
        )
        aspect_threshold = float(
            self.get_parameter("stick_min_aspect_ratio").value
        )
        masks = self.color_masks(rgb)
        any_visible = False
        for color in COLORS:
            components = self.connected_components(
                masks[color], minimum_pixels
            )
            by_kind = {"object": [], "zone": []}
            for component in components:
                aspect = component["height"] / max(1, component["width"])
                kind = "zone" if aspect >= aspect_threshold else "object"
                by_kind[kind].append(component)

            visible_components = []
            for kind in KINDS:
                visible = self._update_shape_visibility(
                    color, kind, bool(by_kind[kind])
                )
                distance = self.target_distance(color, kind)
                reached = (
                    visible
                    and np.isfinite(distance)
                    and distance
                    <= float(self.get_parameter("reached_distance").value)
                )
                suffix = (
                    "reached"
                    if reached
                    else "visible"
                    if visible
                    else "not_visible"
                )
                self.event_publisher.publish(
                    String(data=f"EV_{color}_{kind}_{suffix}")
                )
                any_visible = any_visible or visible
                if by_kind[kind]:
                    visible_components.extend(by_kind[kind])
                    component = max(
                        by_kind[kind], key=lambda item: item["pixels"]
                    )
                    kind_offset = float(
                        np.clip(
                            (2.0 * component["center_x"]
                             / max(1, rgb.shape[1] - 1)) - 1.0,
                            -1.0,
                            1.0,
                        )
                    )
                    self.shape_offset_publishers[(color, kind)].publish(
                        Float32(data=kind_offset)
                    )

            # Preserve the steering interface consumed by TargetApproachMixin.
            if visible_components:
                component = max(
                    visible_components, key=lambda item: item["pixels"]
                )
                offset = float(
                    np.clip(
                        (2.0 * component["center_x"]
                         / max(1, rgb.shape[1] - 1)) - 1.0,
                        -1.0,
                        1.0,
                    )
                )
                self.horizontal_offset_publishers[color].publish(
                    Float32(data=offset)
                )
        self.any_color_visible_publisher.publish(Bool(data=any_visible))


def main(args=None):
    rclpy.init(args=args)
    node = DeliveryShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
