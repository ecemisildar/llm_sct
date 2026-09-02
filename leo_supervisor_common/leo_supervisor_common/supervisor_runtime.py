"""Shared, parameter-neutral runtime methods for all robot supervisors."""

from __future__ import annotations

import math
import os
import re
import time
from typing import Optional

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import String

from leo_supervisor_common.decision_logger import SctDecisionLogger
from leo_supervisor_common.motion import (
    ActionSpec,
    wrap_to_pi as _wrap_to_pi,
    yaw_from_quat as _yaw_from_quat,
)


class SupervisorRuntimeMixin:
    """Common execution helpers relying on attributes initialized by each mission.

    Mission constructors retain ownership of mission-specific parameters while
    this mixin initializes the motion, sensing, recovery, and logging runtime.
    """

    def _initialize_runtime_parameters(self, motion_hold_default: float):
        self.supervisor_period = float(
            self.declare_parameter("supervisor_period", 0.1).value
        )
        self.motion_hold_duration = float(
            self.declare_parameter(
                "motion_hold_duration", motion_hold_default
            ).value
        )
        self.full_rotate_target_rad = min(
            math.pi,
            float(
                self.declare_parameter(
                    "full_rotate_target_rad", math.pi
                ).value
            ),
        )
        self.full_rotate_omega = float(
            self.declare_parameter("full_rotate_omega", 1.0).value
        )
        self.full_rotate_timeout_s = float(
            self.declare_parameter("full_rotate_timeout_s", 3.5).value
        )
        self.post_turn_settle_s = self.motion_hold_duration
        self.recovery_back_hold_s = self.motion_hold_duration
        self.short_rotation_omega = float(
            self.declare_parameter("rotate_90_omega", 1.0).value
        )
        self.zone_update_min_dt = self.supervisor_period
        self.obstacle_zone_memory_s = 4.0 * self.supervisor_period
        self._initialize_zone_escape()
        self.full_rotate_retrigger_block_s = self.escape_cooldown_s

        self.ns = self.get_namespace().strip("/") or "root"
        self.robot_index = self._namespace_index()
        base_seed = int(self.declare_parameter("random_seed", 12345).value)
        self.robot_seed = base_seed + self.robot_index
        self.sct_decision_log_enabled = bool(
            self.declare_parameter("sct_decision_log_enabled", True).value
        )
        self.results_dir = str(
            self.declare_parameter("results_dir", "").value
        ).strip()
        self.run_id = str(self.declare_parameter("run_id", "").value).strip()
        self.total_robots = int(
            self.declare_parameter("total_robots", 1).value
        )
        self.contact_recovery_enabled = bool(
            self.declare_parameter("contact_recovery_enabled", True).value
        )
        self.contact_recovery_duration_s = float(
            self.declare_parameter("contact_recovery_duration_s", 1.0).value
        )
        self.contact_recovery_linear_x = float(
            self.declare_parameter("contact_recovery_linear_x", -0.2).value
        )
        self.contact_recovery_angular_z = float(
            self.declare_parameter("contact_recovery_angular_z", 2.0).value
        )
        self.contact_recovery_retrigger_block_s = (
            self.contact_recovery_duration_s
        )
        self.contact_topic = str(
            self.declare_parameter(
                "contact_topic",
                f"/world/random_world/model/{self.ns}/link/{self.ns}/base_footprint/sensor/contact_sensor/contact",
            ).value
        )

    def _initialize_runtime_state(self, snapshot_columns=()):
        self.obstacle_zones = ["CLEAR"]
        self.last_non_clear_obstacle_zone = "CLEAR"
        self.last_non_clear_obstacle_zone_time = 0.0
        self.last_zone_update = 0.0
        self.last_logged_zone = "CLEAR"
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_stamp_sec = 0
        self.odom_stamp_nsec = 0
        self.active_event = None
        self.active_twist = Twist()
        self.motion_until = 0.0
        self.full_rotate_active = False
        self.full_rotate_accum = 0.0
        self.full_rotate_started_at = 0.0
        self.full_rotate_prev_yaw = 0.0
        self.last_full_rotate_completed_at = 0.0
        self.turn_settle_until = 0.0
        self.contact_recovery_until = 0.0
        self.last_contact_recovery_started_at = 0.0
        self.contact_recovery_source = ""
        self.contact_recovery_turn_sign = (
            1.0 if self.robot_index % 2 == 0 else -1.0
        )
        self.supervisor_started_at = time.time()
        self.decision_logger = SctDecisionLogger(
            enabled=self.sct_decision_log_enabled,
            robot=self.ns,
            results_dir=self.results_dir,
            yaml_path=self.explicit_yaml_path,
            total_robots=self.total_robots,
            run_id=self.run_id,
            snapshot_columns=snapshot_columns,
        )

    def _create_runtime_interfaces(self):
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.executed_event_pub = self.create_publisher(
            String, "executed_event", 10
        )
        self.sub_zone = self.create_subscription(
            String, "detected_zones", self.zone_callback, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )
        self.sub_contact = self.create_subscription(
            Contacts, self.contact_topic, self.contact_callback, 10
        )
        self.timer = self.create_timer(
            self.supervisor_period, self.timer_callback
        )

    def _initialize_motion_action_table(self):
        self.action_table = {
            "EV_move_forward": ActionSpec(linear_x=0.3, angular_z=0.0),
            "EV_move_backward": ActionSpec(
                linear_x=-0.2,
                angular_z=0.0,
                hold_s=self.recovery_back_hold_s,
            ),
            "EV_rotate_clockwise": ActionSpec(
                angular_z=-self.short_rotation_omega,
                hold_s=self.supervisor_period,
            ),
            "EV_rotate_counterclockwise": ActionSpec(
                angular_z=self.short_rotation_omega,
                hold_s=self.supervisor_period,
            ),
            "EV_full_rotate": ActionSpec(
                angular_z=self.full_rotate_omega, is_full_rotate=True
            ),
        }

    def _load_named_sct(self, default_filename: str):
        config_path = self.explicit_yaml_path or os.path.join(
            self.config_dir, default_filename
        )
        if not os.path.exists(config_path):
            label = "Explicit supervisor YAML" if self.explicit_yaml_path else "Supervisor YAML"
            self.get_logger().error(f"{label} not found: {config_path}")
            raise SystemExit(1)
        self._load_sct_from_yaml(config_path)
        source = "explicit YAML " if self.explicit_yaml_path else ""
        self.get_logger().info(
            f"Loaded initial mission '{self.current_mission}' from "
            f"{source}{os.path.basename(config_path)}"
        )

    def zone_callback(self, msg: String):
        now = time.time()
        if now - self.last_zone_update < self.zone_update_min_dt:
            return
        self.last_zone_update = now
        zone = msg.data.strip().upper()
        if zone not in {"LEFT", "RIGHT", "CORNER", "CLEAR"}:
            zone = "CLEAR"
        self.obstacle_zones = [zone]
        if zone != "CLEAR":
            self.last_non_clear_obstacle_zone = zone
            self.last_non_clear_obstacle_zone_time = now
        if zone in {"LEFT", "RIGHT"}:
            self._record_zone_for_livelock(now, zone)
        if zone != self.last_logged_zone:
            if getattr(self, "log_zone_changes", False):
                self.get_logger().info(f"Detected zone changed to {zone}")
            self.last_logged_zone = zone

    def _contact_recovery_blocked(self) -> bool:
        return False

    def contact_callback(self, msg: Contacts):
        if (
            self._contact_recovery_blocked()
            or not self.contact_recovery_enabled
            or not msg.contacts
        ):
            return
        now = time.time()
        if (
            now - self.last_contact_recovery_started_at
            < self.contact_recovery_retrigger_block_s
        ):
            return
        self.last_contact_recovery_started_at = now
        self.contact_recovery_until = max(
            self.contact_recovery_until,
            now + self.contact_recovery_duration_s,
        )
        other_robot_index = self._contact_robot_index(msg)
        if other_robot_index is None:
            self.contact_recovery_turn_sign = (
                1.0 if self.robot_index % 2 == 0 else -1.0
            )
        elif self.robot_index < other_robot_index:
            self.contact_recovery_turn_sign = 1.0
        else:
            self.contact_recovery_turn_sign = -1.0
        self.contact_recovery_source = self._contact_summary(msg)
        self.full_rotate_active = False
        self.active_event = None
        self.motion_until = 0.0
        self.get_logger().info(
            "CONTACT RECOVERY activated "
            f"(source={self.contact_recovery_source}, "
            f"duration_s={self.contact_recovery_duration_s:.2f})"
        )

    def _publish_cmd(self, twist: Twist):
        self.executed_event_pub.publish(
            String(data=self._current_command_event_label())
        )
        self.cmd_pub.publish(twist)

    def destroy_node(self):
        self.decision_logger.write_summary()
        super().destroy_node()

    def _current_command_event_label(self) -> str:
        if self.contact_recovery_until > time.time():
            return "CONTACT_RECOVERY"
        if self.escape_phase is not None:
            return f"ZONE_ESCAPE_{self.escape_phase.upper()}"
        if self.full_rotate_active:
            return self.active_event or "FULL_ROTATE"
        return self.active_event or "none"

    def odom_callback(self, msg: Odometry):
        self.have_odom = True
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.odom_stamp_sec = int(msg.header.stamp.sec)
        self.odom_stamp_nsec = int(msg.header.stamp.nanosec)

    def _effective_obstacle_zones(self):
        zones = []
        if self.obstacle_zones[0] != "CLEAR":
            zones.extend(self.obstacle_zones)
        elif (
            self.last_non_clear_obstacle_zone != "CLEAR"
            and (time.time() - self.last_non_clear_obstacle_zone_time)
            <= self.obstacle_zone_memory_s
        ):
            zones.append(self.last_non_clear_obstacle_zone)
        if not zones:
            return ["CLEAR"]
        unique_zones = []
        for zone in zones:
            if zone not in unique_zones:
                unique_zones.append(zone)
        return unique_zones

    def _contact_robot_index(self, msg: Contacts) -> Optional[int]:
        for contact in msg.contacts:
            for collision in (contact.collision1, contact.collision2):
                name = getattr(collision, "name", "")
                match = re.search(r"robot_(\d+)", name)
                if match:
                    index = int(match.group(1))
                    if index != self.robot_index:
                        return index
        return None

    def _contact_summary(self, msg: Contacts) -> str:
        names = []
        for contact in msg.contacts[:3]:
            for collision in (contact.collision1, contact.collision2):
                name = getattr(collision, "name", str(collision))
                if self.ns not in name:
                    names.append(name)
        return ",".join(names[:3]) or "local_contact"

    def clear_path_check(self, sup_data):
        depth_obstacles = {"LEFT", "RIGHT", "CORNER"}
        return not any(
            zone in depth_obstacles for zone in self._effective_obstacle_zones()
        )

    def middle_check(self, sup_data):
        return "CORNER" in self._effective_obstacle_zones()

    def left_check(self, sup_data):
        return "LEFT" in self._effective_obstacle_zones()

    def right_check(self, sup_data):
        return "RIGHT" in self._effective_obstacle_zones()

    def _namespace_index(self) -> int:
        if self.ns.startswith("robot_"):
            try:
                return int(self.ns.split("_")[-1])
            except ValueError:
                return 0
        return 0

    def _publish_stop(self):
        self.active_twist = Twist()
        self._publish_cmd(self.active_twist)

    def _publish_contact_recovery_cmd(self):
        twist = Twist()
        linear_x = self.contact_recovery_linear_x
        turn_sign = self.contact_recovery_turn_sign
        zones = self._effective_obstacle_zones()
        if "CORNER" in zones:
            linear_x = min(linear_x, 0.0)
        elif "BACK" in zones:
            linear_x = 0.0
        elif "LEFT" in zones:
            turn_sign = -1.0
        elif "RIGHT" in zones:
            turn_sign = 1.0
        twist.linear.x = linear_x
        twist.angular.z = turn_sign * abs(self.contact_recovery_angular_z)
        self.active_twist = twist
        self._publish_cmd(self.active_twist)

    def _enter_post_turn_settle(self, now: float):
        self.turn_settle_until = max(
            self.turn_settle_until, now + self.post_turn_settle_s
        )
        self.active_event = None
        self.motion_until = 0.0
        self._publish_stop()

    def _start_full_rotate(self, omega: float):
        self.full_rotate_active = True
        self.full_rotate_started_at = time.time()
        self.full_rotate_accum = 0.0
        if self.have_odom:
            self.full_rotate_prev_yaw = self.yaw
        else:
            dur = abs(self.full_rotate_target_rad / max(1e-6, abs(omega)))
            self.motion_until = time.time() + min(
                dur, self.full_rotate_timeout_s
            )
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = omega
        self.active_twist = twist
        self.get_logger().info(
            "Starting FULL ROTATE "
            f"(omega={omega:.3f}, have_odom={self.have_odom}, "
            f"target_rad={self.full_rotate_target_rad:.3f}, "
            f"timeout_s={self.full_rotate_timeout_s:.3f})"
        )
        self._publish_cmd(self.active_twist)

    def _update_full_rotate(self) -> bool:
        now = time.time()
        if (now - self.full_rotate_started_at) > self.full_rotate_timeout_s:
            self.get_logger().info("FULL ROTATE completed by timeout")
            return True
        if not self.have_odom:
            done = now >= self.motion_until
            if done:
                self.get_logger().info("FULL ROTATE completed by timed fallback")
            return done
        dy = _wrap_to_pi(self.yaw - self.full_rotate_prev_yaw)
        self.full_rotate_accum += abs(dy)
        self.full_rotate_prev_yaw = self.yaw
        done = self.full_rotate_accum >= self.full_rotate_target_rad
        if done:
            self.get_logger().info(
                "FULL ROTATE completed by odom "
                f"(accum_rad={self.full_rotate_accum:.3f})"
            )
        return done
