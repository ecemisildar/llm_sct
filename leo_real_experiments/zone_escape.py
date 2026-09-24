"""Shared obstacle-zone and no-progress livelock recovery."""

import math
from collections import deque

from geometry_msgs.msg import Twist


class ZoneLivelockEscapeMixin:
    """Detect oscillation or no progress and execute an atomic escape."""

    def _initialize_zone_escape(self):
        parameter_defaults = {
            "zone_livelock_window_s": 2.5,
            "zone_livelock_switches": 4,
            "zone_livelock_max_displacement_m": 0.12,
            "motion_livelock_window_s": 3.0,
            "motion_livelock_switches": 4,
            "motion_stuck_timeout_s": 4.0,
            "escape_reverse_s": 0.6,
            "escape_turn_s": 1.8,
            "escape_linear_x": -0.2,
            "escape_angular_z": 1.2,
            "escape_cooldown_s": 3.0,
        }
        for name, default in parameter_defaults.items():
            value = self.declare_parameter(name, default).value
            setattr(self, name, type(default)(value))
        self.zone_switch_history = deque()
        self.turn_switch_history = deque()
        self.motion_progress_started_at = None
        self.motion_progress_anchor_x = 0.0
        self.motion_progress_anchor_y = 0.0
        self.escape_phase = None
        self.escape_phase_until = 0.0
        self.escape_turn_sign = 1.0
        self.escape_cooldown_until = 0.0

    def _start_livelock_escape(self, now: float, turn_sign: float, reason: str):
        self.escape_turn_sign = turn_sign
        self.escape_phase = "reverse"
        self.escape_phase_until = now + self.escape_reverse_s
        self.zone_switch_history.clear()
        self.turn_switch_history.clear()
        self.motion_progress_started_at = None
        self.full_rotate_active = False
        self.active_event = None
        self.motion_until = 0.0
        self.get_logger().debug(
            f"{reason} livelock detected; starting atomic escape "
            f"(turn_sign={turn_sign:+.0f})"
        )

    def _record_zone_for_livelock(self, now: float, zone: str):
        if self.escape_phase is not None or now < self.escape_cooldown_until:
            return
        if self.zone_switch_history and self.zone_switch_history[-1][1] == zone:
            return
        self.zone_switch_history.append((now, zone, self.x, self.y))
        cutoff = now - self.zone_livelock_window_s
        while self.zone_switch_history and self.zone_switch_history[0][0] < cutoff:
            self.zone_switch_history.popleft()
        if len(self.zone_switch_history) < self.zone_livelock_switches + 1:
            return
        first = self.zone_switch_history[0]
        displacement = math.hypot(self.x - first[2], self.y - first[3])
        if self.have_odom and displacement > self.zone_livelock_max_displacement_m:
            return
        sign = -1.0 if zone == "LEFT" else 1.0
        self._start_livelock_escape(
            now, sign, f"LEFT/RIGHT zone (displacement_m={displacement:.3f})"
        )

    def _record_motion_for_livelock(
        self, now: float, linear_x: float, angular_z: float
    ) -> bool:
        if self.escape_phase is not None or now < self.escape_cooldown_until:
            return self.escape_phase is not None

        # Displacement-based recovery is meaningless until a real odometry
        # sample has arrived.  Previously x=y=0 caused moving robots to be
        # classified as stuck when their configured odom topic was absent.
        if not self.have_odom:
            self.motion_progress_started_at = None
            return False

        if self.motion_progress_started_at is None:
            self.motion_progress_started_at = now
            self.motion_progress_anchor_x = self.x
            self.motion_progress_anchor_y = self.y
        progress = math.hypot(
            self.x - self.motion_progress_anchor_x,
            self.y - self.motion_progress_anchor_y,
        )
        if progress > self.zone_livelock_max_displacement_m:
            self.motion_progress_started_at = now
            self.motion_progress_anchor_x = self.x
            self.motion_progress_anchor_y = self.y
            self.turn_switch_history.clear()

        zones = self._effective_obstacle_zones()
        side_blocked = any(z in {"LEFT", "RIGHT", "CORNER"} for z in zones)
        stalled_for = now - self.motion_progress_started_at
        is_translation = abs(linear_x) > 0.05
        if abs(angular_z) > 0.05:
            sign = 1.0 if angular_z > 0.0 else -1.0
        else:
            # Neighboring robots choose opposite escape turns when they meet.
            sign = 1.0 if getattr(self, "robot_index", 1) % 2 else -1.0
        if (
            side_blocked
            and is_translation
            and stalled_for >= self.motion_stuck_timeout_s
        ):
            self._start_livelock_escape(
                now, sign, f"no translation for {stalled_for:.1f}s"
            )
            return True

        # A deliberate in-place turn cannot produce translation. Pause the
        # translation-stall clock so it does not trigger on the next command.
        if not is_translation:
            self.motion_progress_started_at = now
            self.motion_progress_anchor_x = self.x
            self.motion_progress_anchor_y = self.y

        if abs(linear_x) > 0.05 or abs(angular_z) < 0.05:
            return False
        if self.turn_switch_history and self.turn_switch_history[-1][1] == sign:
            return False
        self.turn_switch_history.append((now, sign, self.x, self.y))
        cutoff = now - self.motion_livelock_window_s
        while self.turn_switch_history and self.turn_switch_history[0][0] < cutoff:
            self.turn_switch_history.popleft()
        if len(self.turn_switch_history) < self.motion_livelock_switches + 1:
            return False
        first = self.turn_switch_history[0]
        displacement = math.hypot(self.x - first[2], self.y - first[3])
        if self.have_odom and displacement > self.zone_livelock_max_displacement_m:
            return False
        self._start_livelock_escape(
            now, sign, f"alternating turn command (displacement_m={displacement:.3f})"
        )
        return True

    def _run_zone_escape(self, now: float):
        twist = Twist()
        if self.escape_phase == "reverse":
            if now < self.escape_phase_until:
                twist.linear.x = self.escape_linear_x
                self.active_twist = twist
                self._publish_cmd(twist)
                return
            self.escape_phase = "turn"
            self.escape_phase_until = now + self.escape_turn_s
        if now < self.escape_phase_until:
            twist.angular.z = self.escape_turn_sign * abs(self.escape_angular_z)
            self.active_twist = twist
            self._publish_cmd(twist)
            return
        self.escape_phase = None
        self.escape_cooldown_until = now + self.escape_cooldown_s
        self._enter_post_turn_settle(now)
        self.get_logger().info("Zone-livelock escape completed")
