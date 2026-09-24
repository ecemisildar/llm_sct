"""Shared odometry-based command filter with stuck escape recovery."""

import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


TARGET_COLOR_ORDER = ("red", "green", "blue")


class StuckRecovery(Node):
    """Forward supervisor velocity commands and override them during recovery."""

    def __init__(self):
        super().__init__("stuck_recovery")
        self.enabled = bool(self.declare_parameter("enabled", True).value)
        self.timeout_s = max(
            1.0, float(self.declare_parameter("timeout_s", 5.0).value)
        )
        self.displacement_m = max(
            0.01, float(self.declare_parameter("displacement_m", 0.10).value)
        )
        self.escape_turn_rad = min(
            math.pi,
            max(
                math.pi / 4.0,
                float(
                    self.declare_parameter(
                        "escape_turn_rad", 3.0 * math.pi / 4.0
                    ).value
                ),
            ),
        )
        self.escape_angular_z = max(
            0.1, abs(float(self.declare_parameter("escape_angular_z", 1.2).value))
        )
        self.reverse_linear_x = -abs(
            float(self.declare_parameter("reverse_linear_x", -0.2).value)
        )
        self.reverse_duration_s = max(
            0.1, float(self.declare_parameter("reverse_duration_s", 0.8).value)
        )
        self.forward_linear_x = abs(
            float(self.declare_parameter("forward_linear_x", 0.3).value)
        )
        self.forward_duration_s = max(
            0.1, float(self.declare_parameter("forward_duration_s", 0.8).value)
        )
        self.cooldown_s = max(
            0.0, float(self.declare_parameter("cooldown_s", 3.0).value)
        )
        self.command_timeout_s = max(
            0.1, float(self.declare_parameter("command_timeout_s", 1.5).value)
        )
        self.cmd_vel_topic = str(
            self.declare_parameter("cmd_vel_topic", "cmd_vel").value
        )
        self.supervisor_cmd_vel_topic = str(
            self.declare_parameter(
                "supervisor_cmd_vel_topic", "cmd_vel_supervisor"
            ).value
        )
        self.odom_topic = str(self.declare_parameter("odom_topic", "odom").value)
        self.robot_index = self._robot_index()
        self.latest_command = Twist()
        self.task_completed = False
        self.latest_command_time = 0.0
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.anchor_x: Optional[float] = None
        self.anchor_y: Optional[float] = None
        self.anchor_time = time.time()
        self.obstacle_zone = "CLEAR"
        configured_color_order = str(
            self.declare_parameter(
                "target_color_order", ",".join(TARGET_COLOR_ORDER)
            ).value
        )
        parsed_color_order = tuple(
            color.strip().casefold()
            for color in configured_color_order.split(",")
            if color.strip()
        )
        if (
            not parsed_color_order
            or set(parsed_color_order) - set(TARGET_COLOR_ORDER)
        ):
            self.get_logger().warning(
                f"Invalid target_color_order '{configured_color_order}'; "
                "using red,green,blue."
            )
            parsed_color_order = TARGET_COLOR_ORDER
        self.target_color_order = parsed_color_order
        self.visible_by_color = {
            color: False for color in TARGET_COLOR_ORDER
        }
        self.reached_goal_index = 0
        self.recovery_active = False
        self.recovery_phase = ""
        self.phase_until = 0.0
        self.cooldown_until = 0.0
        self.turn_sign = 1.0
        self.recovery_count = 0

        self.publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(
            Twist, self.supervisor_cmd_vel_topic, self.command_callback, 10
        )
        odom_topics = []
        for topic in (
            self.odom_topic,
            "merged_odom",
            "/merged_odom",
            "odom",
            "/odom",
        ):
            if topic and topic not in odom_topics:
                odom_topics.append(topic)
        self.odom_subscriptions = [
            self.create_subscription(Odometry, topic, self.odom_callback, 10)
            for topic in odom_topics
        ]
        self.create_subscription(String, "detected_zones", self.zone_callback, 10)
        self.create_subscription(Bool, "task_complete", self.task_complete_callback, 10)
        for color in TARGET_COLOR_ORDER:
            self.create_subscription(
                Bool,
                f"{color}_visible",
                lambda message, color=color: self.color_visible_callback(
                    color, message
                ),
                10,
            )
        self.create_subscription(
            String,
            "task_progress_event",
            self.task_progress_callback,
            10,
        )
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(
            "Shared stuck recovery active: cmd_vel_supervisor -> cmd_vel "
            "(reverse -> turn -> forward)"
        )

    def _robot_index(self) -> int:
        namespace = self.get_namespace().strip("/")
        try:
            return int(namespace.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def _command_requests_translation(command: Twist) -> bool:
        return abs(command.linear.x) > 1e-3

    def command_callback(self, command: Twist):
        self.latest_command = command
        self.latest_command_time = time.time()

    def odom_callback(self, message: Odometry):
        self.have_odom = True
        self.x = float(message.pose.pose.position.x)
        self.y = float(message.pose.pose.position.y)
        now = time.time()
        if self.anchor_x is None or self.anchor_y is None:
            self._reset_anchor(now)
            return
        if math.hypot(self.x - self.anchor_x, self.y - self.anchor_y) >= self.displacement_m:
            self._reset_anchor(now)

    def zone_callback(self, message: String):
        zone = message.data.strip().upper()
        self.obstacle_zone = zone if zone in {"LEFT", "RIGHT", "CORNER", "CLEAR"} else "CLEAR"

    def color_visible_callback(self, color: str, message: Bool):
        self.visible_by_color[color] = bool(message.data)

    def task_progress_callback(self, message: String):
        color = message.data.strip().lower()
        if (
            self.reached_goal_index < len(self.target_color_order)
            and color == self.target_color_order[self.reached_goal_index]
        ):
            self.reached_goal_index += 1

    def _expected_color(self) -> Optional[str]:
        if self.reached_goal_index >= len(self.target_color_order):
            return None
        return self.target_color_order[self.reached_goal_index]

    def _expected_color_visible(self) -> bool:
        expected = self._expected_color()
        return expected is not None and self.visible_by_color[expected]

    def _reset_anchor(self, now: float):
        self.anchor_x = self.x
        self.anchor_y = self.y
        self.anchor_time = now

    def _start_recovery(self, now: float):
        if self.obstacle_zone == "LEFT":
            self.turn_sign = -1.0
        elif self.obstacle_zone == "RIGHT":
            self.turn_sign = 1.0
        else:
            self.turn_sign = (
                1.0 if (self.robot_index + self.recovery_count) % 2 == 0 else -1.0
            )
        self.recovery_count += 1
        self.recovery_active = True
        self.recovery_phase = "reverse"
        self.phase_until = now + self.reverse_duration_s
        self.get_logger().warning(
            "STUCK RECOVERY activated "
            f"(stationary_s={now - self.anchor_time:.2f}, "
            f"sequence=reverse->turn->forward, "
            f"turn_rad={self.escape_turn_rad:.2f}, zone={self.obstacle_zone})"
        )

    def _cancel_recovery_for_target(self, now: float):
        expected = self._expected_color()
        self.recovery_active = False
        self.recovery_phase = ""
        self.cooldown_until = now + self.cooldown_s
        self._reset_anchor(now)
        self.publisher.publish(
            self.latest_command
            if (now - self.latest_command_time) <= self.command_timeout_s
            else Twist()
        )
        self.get_logger().info(
            f"STUCK RECOVERY cancelled because expected color '{expected}' is visible."
        )

    def _advance_recovery_phase(self, now: float):
        if self.recovery_phase == "reverse":
            self.recovery_phase = "turn"
            self.phase_until = (
                now + self.escape_turn_rad / self.escape_angular_z
            )
            return
        if self.recovery_phase == "turn":
            self.recovery_phase = "forward"
            self.phase_until = now + self.forward_duration_s
            return

        self.recovery_active = False
        self.recovery_phase = ""
        self.cooldown_until = now + self.cooldown_s
        self._reset_anchor(now)
        self.publisher.publish(Twist())
        self.get_logger().info("STUCK RECOVERY completed; resuming supervisor.")

    def _publish_recovery_command(self):
        command = Twist()
        if self.recovery_phase == "reverse":
            command.linear.x = self.reverse_linear_x
        elif self.recovery_phase == "turn":
            command.angular.z = self.turn_sign * self.escape_angular_z
        elif self.recovery_phase == "forward":
            command.linear.x = self.forward_linear_x
        self.publisher.publish(command)

    def task_complete_callback(self, message: Bool):
        if message.data:
            self.task_completed = True
            self.recovery_active = False
            self.recovery_phase = ""
            self.latest_command = Twist()
            self.publisher.publish(Twist())

    def timer_callback(self):
        if self.task_completed:
            self.publisher.publish(Twist())
            return
        now = time.time()
        command_is_fresh = (now - self.latest_command_time) <= self.command_timeout_s
        commanded_translation = command_is_fresh and self._command_requests_translation(
            self.latest_command
        )

        if self.recovery_active:
            if self._expected_color_visible():
                self._cancel_recovery_for_target(now)
                return
            if now >= self.phase_until:
                self._advance_recovery_phase(now)
                if not self.recovery_active:
                    return
            self._publish_recovery_command()
            return

        if not commanded_translation:
            # Rotation-only commands are intentional search behavior, not
            # evidence that the robot is physically stuck.
            self._reset_anchor(now)

        if (
            self.enabled
            and self.have_odom
            and not self._expected_color_visible()
            and commanded_translation
            and now >= self.cooldown_until
            and (now - self.anchor_time) >= self.timeout_s
        ):
            self._start_recovery(now)
            return

        self.publisher.publish(self.latest_command if command_is_fresh else Twist())


def main(args=None):
    rclpy.init(args=args)
    node = StuckRecovery()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
