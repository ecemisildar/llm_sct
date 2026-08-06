"""Shared, parameter-neutral runtime methods for all robot supervisors."""

from __future__ import annotations

import re
import time
from typing import Optional

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import String

from leo_supervisor_common.motion import (
    wrap_to_pi as _wrap_to_pi,
    yaw_from_quat as _yaw_from_quat,
)


class SupervisorRuntimeMixin:
    """Common execution helpers relying on attributes initialized by each mission.

    This mixin deliberately declares no ROS parameters. Mission constructors retain
    ownership of every parameter name, default, and interpretation.
    """

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
        hit = "CORNER" in self._effective_obstacle_zones()
        if hit and not self.front_obstacle_active:
            now = time.time()
            self.block_full_rotate_until = max(
                self.block_full_rotate_until,
                now + self.full_rotate_block_after_obs_front,
            )
        self.front_obstacle_active = hit
        return hit

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
        self.full_rotate_using_timed_fallback = False
        self.full_rotate_stall_count = 0
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
        if not self.have_odom or self.full_rotate_using_timed_fallback:
            done = now >= self.motion_until
            if done:
                self.get_logger().info("FULL ROTATE completed by timed fallback")
            return done
        dy = _wrap_to_pi(self.yaw - self.full_rotate_prev_yaw)
        self.full_rotate_accum += abs(dy)
        self.full_rotate_prev_yaw = self.yaw
        if abs(dy) < 1e-3:
            self.full_rotate_stall_count += 1
        else:
            self.full_rotate_stall_count = 0
        if self.full_rotate_stall_count >= 5:
            remaining = max(
                0.0, self.full_rotate_target_rad - self.full_rotate_accum
            )
            omega = max(1e-6, abs(self.active_twist.angular.z))
            dur = min(remaining / omega, self.full_rotate_timeout_s)
            self.motion_until = now + dur
            self.full_rotate_using_timed_fallback = True
            self.get_logger().info(
                "FULL ROTATE switching to timed fallback "
                f"(accum_rad={self.full_rotate_accum:.3f}, "
                f"remaining_rad={remaining:.3f}, "
                f"stall_count={self.full_rotate_stall_count}, "
                f"fallback_s={dur:.3f})"
            )
        done = self.full_rotate_accum >= self.full_rotate_target_rad
        if done:
            self.get_logger().info(
                "FULL ROTATE completed by odom "
                f"(accum_rad={self.full_rotate_accum:.3f})"
            )
        return done
