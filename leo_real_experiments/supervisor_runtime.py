"""Shared, parameter-neutral runtime methods for all robot supervisors."""

from __future__ import annotations

import re
import time
from typing import Optional

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import SetBool

from leo_real_experiments.motion import (
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

    def _create_odom_subscriptions(self):
        """Subscribe to configured and common real-Leo odometry names."""
        topics = []
        for topic in (self.odom_topic, "merged_odom", "/merged_odom", "odom", "/odom"):
            if topic and topic not in topics:
                topics.append(topic)
        self.odom_subscriptions = [
            self.create_subscription(Odometry, topic, self.odom_callback, 10)
            for topic in topics
        ]
        self.get_logger().info(
            "Odometry inputs: " + ", ".join(topics)
        )

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
    def _initialize_enable_service(self):
        self.enabled = bool(self.declare_parameter("enabled", True).value)
        self.enable_service = self.create_service(
            SetBool, "/enable_supervisor", self._set_enabled
        )

    def _set_enabled(self, request, response):
        was_enabled = self.enabled
        self.enabled = bool(request.data)
        if was_enabled and not self.enabled:
            self.active_event = None
            self.motion_until = 0.0
            self.full_rotate_active = False
            self.escape_phase = None
            self._publish_stop()
        response.success = True
        response.message = "supervisor enabled" if self.enabled else "supervisor disabled"
        return response
