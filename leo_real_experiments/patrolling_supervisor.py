import os
import json
import random
import math
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Int32, String
from nav_msgs.msg import Odometry

from ament_index_python.packages import get_package_share_directory
from leo_real_experiments.patrolling_sct import SCT
from leo_real_experiments.decision_logger import SctDecisionLogger
from leo_real_experiments.motion import (
    ActionSpec,
    wrap_to_pi as _wrap_to_pi,
    yaw_from_quat as _yaw_from_quat,
)
from leo_real_experiments.target_approach import TargetApproachMixin
from leo_real_experiments.zone_escape import ZoneLivelockEscapeMixin
from leo_real_experiments.supervisor_runtime import SupervisorRuntimeMixin


DEFAULT_TARGET_COLOR_ORDER = ("red", "green", "blue")


class RobotSupervisor(
    SupervisorRuntimeMixin, TargetApproachMixin, ZoneLivelockEscapeMixin, Node
):
    def __init__(self):
        super().__init__("robot_supervisor")
        self.get_logger().set_level(LoggingSeverity.WARN)

        # -------------------------------
        # Parameters
        # -------------------------------
        self.supervisor_period = float(self.declare_parameter("supervisor_period", 0.1).value)
        self.motion_hold_duration = float(self.declare_parameter("motion_hold_duration", 0.2).value)
        self._initialize_enable_service()
        self.completion_retreat_duration_s = max(
            0.1,
            float(
                self.declare_parameter("completion_retreat_duration_s", 1.5).value
            ),
        )
        self.completion_shutdown_timer = None
        self.task_complete_sent = False
        self.completion_parking_reason = ""
        self.completion_parking_phase = None
        self.completion_parking_until = 0.0
        self.completion_parking_started_at = 0.0
        self.completion_parking_blocked_since = None
        self.completion_parking_search_angular_z = 0.35
        self.completion_parking_timeout_s = max(
            5.0,
            float(
                self.declare_parameter(
                    "completion_parking_timeout_s", 60.0
                ).value
            ),
        )
        self.completion_parking_turn_angular_z = max(0.1, abs(
            float(
                self.declare_parameter(
                    "completion_parking_turn_angular_z", 1.0
                ).value
            )
        ))
        self.completion_parking_stop_distance_m = max(
            0.1,
            float(
                self.declare_parameter(
                    "completion_parking_stop_distance_m", 0.50
                ).value
            ),
        )
        self.parking_wall_distance = float("nan")
        self.parking_wall_updated_at = 0.0

        # Full-rotate execution settings
        # Cap at 180 degrees max, even if overridden via parameters.
        self.full_rotate_target_rad = min(
            math.pi,
            float(self.declare_parameter("full_rotate_target_rad", math.pi).value),
        )
        self.full_rotate_omega = float(self.declare_parameter("full_rotate_omega", 1.0).value)  # rad/s
        self.full_rotate_timeout_s = float(self.declare_parameter("full_rotate_timeout_s", 3.5).value)
        self.full_rotate_retrigger_block_s = float(
            self.declare_parameter("full_rotate_retrigger_block_s", 0.6).value
        )
        self.post_turn_settle_s = float(
            self.declare_parameter("post_turn_settle_s", 0.3).value
        )

        # Obstacle-front escape bias: block full_rotate briefly after obstacle_front
        self.full_rotate_block_after_obs_front = float(
            self.declare_parameter("full_rotate_block_after_obs_front", 0.8).value
        )  # seconds
        self.block_full_rotate_until = 0.0
        self.front_obstacle_active = False
        self.recovery_back_hold_s = float(
            self.declare_parameter("recovery_back_hold_s", 0.35).value
        )

        # Angular speed for one-tick directional turns.
        self.short_rotation_omega = float(
            self.declare_parameter("rotate_90_omega", 1.0).value
        )


        # Zone update throttle. Keep low for fast reaction to depth obstacles.
        self.zone_update_min_dt = float(
            self.declare_parameter("zone_update_min_dt", 0.1).value
        )
        self.obstacle_zone_memory_s = float(
            self.declare_parameter("obstacle_zone_memory_s", 0.4).value
        )
        self._initialize_zone_escape()

        # Random seed (per-robot namespace)
        self.ns = self.get_namespace().strip("/") or "root"
        self.robot_index = self._namespace_index()
        self.cmd_vel_topic = str(
            self.declare_parameter("cmd_vel_topic", "cmd_vel").value
        )
        self.odom_topic = str(self.declare_parameter("odom_topic", "odom").value)
        self.forward_linear_x = float(
            self.declare_parameter("forward_linear_x", 0.3).value
        )
        self.backward_linear_x = -abs(
            float(self.declare_parameter("backward_linear_x", -0.2).value)
        )
        self.completion_retreat_linear_x = -abs(
            float(
                self.declare_parameter(
                    "completion_retreat_linear_x", self.backward_linear_x
                ).value
            )
        )
        self.completion_parking_linear_x = abs(
            float(
                self.declare_parameter(
                    "completion_parking_linear_x", 0.20
                ).value
            )
        )

        base_seed = int(self.declare_parameter("random_seed", 12345).value)
        robot_seed = base_seed + self.robot_index
        self.rng = random.Random(robot_seed)
        random.seed(robot_seed)
        self.robot_seed = robot_seed

        
        self.sct_decision_log_enabled = bool(
            self.declare_parameter("sct_decision_log_enabled", True).value
        )
        
        
        self.results_dir = str(self.declare_parameter("results_dir", "").value).strip()
        self.run_id = str(self.declare_parameter("run_id", "").value).strip()
        self.total_robots = int(self.declare_parameter("total_robots", 1).value)
        default_forward_probability = 0.50
        self.forward_probability = float(
            self.declare_parameter(
                "forward_probability",
                default_forward_probability,
            ).value
        )
        self.pending_color_events = set()
        self.deferred_color_events = {}
        self.last_logged_color_events = {}
        configured_color_order = str(
            self.declare_parameter(
                "target_color_order", ",".join(DEFAULT_TARGET_COLOR_ORDER)
            ).value
        )
        parsed_color_order = tuple(
            color.strip().casefold()
            for color in configured_color_order.split(",")
            if color.strip()
        )
        unsupported_colors = (
            set(parsed_color_order) - set(DEFAULT_TARGET_COLOR_ORDER)
        )
        if not parsed_color_order or unsupported_colors:
            self.get_logger().debug(
                f"Invalid target_color_order '{configured_color_order}'; "
                "using red,green,blue."
            )
            parsed_color_order = DEFAULT_TARGET_COLOR_ORDER
        self.target_color_order = parsed_color_order
        self._initialize_target_approach()
        self.reached_color_sequence = []
        self.reached_color_index = 0
        self.reached_color_latches = set()
        self.all_colors_reached = False
        self.get_logger().info(
            "Required color order: " + " -> ".join(self.target_color_order)
        )

        # -------------------------------
        # Load SCT YAML
        # -------------------------------
        self.config_dir = os.path.join(get_package_share_directory("leo_real_experiments"), "config")
        self.explicit_yaml_path = str(self.declare_parameter("supervisor_yaml_path", "").value).strip()
        self.current_mission = "patrolling"
        self.current_yaml_path = ""
        self._load_initial_sct()
        
        self._last_printed_sup_states: Optional[Tuple[int, ...]] = None

        # -------------------------------
        # State (sensing)
        # -------------------------------
        self.obstacle_zones = ["CLEAR"]
        self.last_non_clear_obstacle_zone = "CLEAR"
        self.last_non_clear_obstacle_zone_time = 0.0
        self.last_zone_update = 0.0
        self.last_logged_zone = "CLEAR"
        
        # Odom / yaw tracking for full-rotate
        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_stamp_sec = 0
        self.odom_stamp_nsec = 0

        # -------------------------------
        # State (actuation)
        # -------------------------------
        self.active_event: Optional[str] = None
        self.active_twist = Twist()
        self.motion_until = 0.0

        # Full-rotate mode bookkeeping
        self.full_rotate_active = False
        self.full_rotate_accum = 0.0
        self.full_rotate_started_at = 0.0
        self.full_rotate_prev_yaw = 0.0
        self.full_rotate_using_timed_fallback = False
        self.full_rotate_stall_count = 0
        self.last_full_rotate_completed_at = 0.0
        self.turn_settle_until = 0.0
        self.supervisor_started_at = time.time()
        self.decision_logger = SctDecisionLogger(
            enabled=self.sct_decision_log_enabled,
            robot=self.ns,
            results_dir=self.results_dir,
            yaml_path=self.explicit_yaml_path,
            total_robots=self.total_robots,
            run_id=self.run_id,
        )


        # -------------------------------
        # Publishers/Subscribers
        # -------------------------------
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.executed_event_pub = self.create_publisher(String, "executed_event", 10)
        self.task_complete_pub = self.create_publisher(
            Bool, f"/{self.ns}/task_complete", 10
        )
        self.task_progress_pub = self.create_publisher(Int32, "task_progress", 10)
        self.task_progress_event_pub = self.create_publisher(
            String, "task_progress_event", 10
        )
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.mission_status_pub = self.create_publisher(
            String, "mission_status", status_qos
        )
        self.sub_zone = self.create_subscription(String, "detected_zones", self.zone_callback, 10)
        self.create_subscription(
            String,
            "color_events",
            self.color_event_callback,
            10,
        )
        self.create_subscription(
            Float32, "parking_wall_distance", self._parking_wall_callback, 10
        )
        self._create_target_approach_subscriptions()
        self._create_odom_subscriptions()


        # -------------------------------
        # SCT callbacks for UCEs (data-driven, but still attaches known sensors)
        # -------------------------------
        # Supervisor timer
        self.timer = self.create_timer(self.supervisor_period, self.timer_callback)
        self._last_mission_status = ""
        self._publish_mission_status()

        # -------------------------------
        # Action table (data-driven)
        # -------------------------------
        # Only place you encode motion parameters. No “logic” needs event names elsewhere.
        self.action_table: Dict[str, ActionSpec] = {
            "EV_move_forward": ActionSpec(linear_x=self.forward_linear_x, angular_z=0.0),
            "EV_move_backward": ActionSpec(linear_x=self.backward_linear_x, angular_z=0.0, hold_s=self.recovery_back_hold_s),
            "EV_rotate_clockwise": ActionSpec(
                linear_x=0.0,
                angular_z=-self.short_rotation_omega,
                hold_s=self.supervisor_period,
            ),
            "EV_rotate_counterclockwise": ActionSpec(
                linear_x=0.0,
                angular_z=self.short_rotation_omega,
                hold_s=self.supervisor_period,
            ),
            # full_rotate is executed as an atomic rotation using odom; target is full_rotate_target_rad (≤ π here).
            "EV_full_rotate": ActionSpec(linear_x=0.0, angular_z=self.full_rotate_omega, is_full_rotate=True),
            "EV_u_turn": ActionSpec(linear_x=0.0, angular_z=self.full_rotate_omega, is_full_rotate=True),
        }

    def _load_sct_from_yaml(self, config_path: str):
        self.sct = SCT(
            config_path,
            random_seed=self.robot_seed,
            forward_probability=self.forward_probability,
        )
        self.ev_name_by_id = {ev_id: ev_name for ev_name, ev_id in self.sct.EV.items()}
        for ev_id in self.sct.EV.values():
            if ev_id not in self.sct.callback:
                self.sct.callback[ev_id] = {
                    "callback": None,
                    "check_input": (lambda _sup_data: False),
                    "sup_data": None,
                }
        self._install_uncontrollable_callbacks()

    def _load_initial_sct(self):
        if self.explicit_yaml_path:
            if not os.path.exists(self.explicit_yaml_path):
                self.get_logger().error(
                    f"Explicit supervisor YAML not found: {self.explicit_yaml_path}"
                )
                raise SystemExit(1)
            config_path = self.explicit_yaml_path
            self._load_sct_from_yaml(config_path)
            self.current_yaml_path = config_path
            self.get_logger().info(
                f"Loaded initial mission '{self.current_mission}' from explicit YAML {os.path.basename(config_path)}"
            )
            return
        config_path = os.path.join(self.config_dir, "sup_patrolling.yaml")
        if not os.path.exists(config_path):
            self.get_logger().error(
                f"Supervisor YAML not found: {config_path}"
            )
            raise SystemExit(1)
        self._load_sct_from_yaml(config_path)
        self.current_yaml_path = config_path
        self.get_logger().info(
            f"Loaded initial mission '{self.current_mission}' from {os.path.basename(config_path)}"
        )

    def _snapshot_sct_inputs(self, now: float):
        effective_zones = self._effective_obstacle_zones()
        depth_obstacles = {"LEFT", "RIGHT", "CORNER"}
        return {
            "elapsed_s": now - self.supervisor_started_at,
            "raw_zone": self.obstacle_zones[0],
            "effective_zone": "|".join(effective_zones),
            "path_clear": not any(zone in depth_obstacles for zone in effective_zones),
            "obstacle_front": "CORNER" in effective_zones,
            "obstacle_left": "LEFT" in effective_zones,
            "obstacle_right": "RIGHT" in effective_zones,
            "turn_settle_active": now < self.turn_settle_until,
            "active_event_before": self.active_event or "",
        }

    def _log_sct_decision(self, selected_event: str, ce_exists: bool, snapshot):
        uncontrollable_events = [
            self.ev_name_by_id.get(int(event), f"unknown:{int(event)}")
            for event in self.sct.last_uncontrollable_events
        ]
        for event in uncontrollable_events:
            if re.fullmatch(
                r"EV_(red|green|blue)_(not_visible|visible|reached)", event
            ):
                continue
            self.get_logger().warning(
                f"Triggered uncontrollable event: {event}"
            )
        if ce_exists and selected_event not in {"none", ""} and not selected_event.startswith("unknown:"):
            self.get_logger().warning(
                f"Triggered controllable event: {selected_event}"
            )

        self.decision_logger.log(
            selected_event, ce_exists, uncontrollable_events, snapshot
        )

    # -------------------------------
    # Subscriptions
    # -------------------------------
    def _motion_event_in_progress(self) -> bool:
        """Return whether a selected motion event still owns the robot."""
        now = time.time()
        return (
            self.full_rotate_active
            or now < self.turn_settle_until
            or (self.active_event is not None and now < self.motion_until)
        )

    def color_event_callback(self, msg):
        event = msg.data.strip()
        color_events = {
            "EV_red_not_visible",
            "EV_red_visible",
            "EV_red_reached",
            "EV_green_not_visible",
            "EV_green_visible",
            "EV_green_reached",
            "EV_blue_not_visible",
            "EV_blue_visible",
            "EV_blue_reached",
        }
        if event not in color_events:
            return

        color = event.split("_")[1]
        if self._motion_event_in_progress():
            # Retain only the newest observation for this color. It will not be
            # presented to the SCT until the executing action has finished.
            self.deferred_color_events[color] = event
            return

        self._accept_color_event(event)

    def _accept_color_event(self, event: str):
        """Queue a color event when no motion event is executing."""
        color = event.split("_")[1]
        if self.last_logged_color_events.get(color) != event:
            self.last_logged_color_events[color] = event
            self.get_logger().warning(
                f"Triggered uncontrollable event: {event}"
            )
        self.pending_color_events.difference_update(
            {
                f"EV_{color}_not_visible",
                f"EV_{color}_visible",
                f"EV_{color}_reached",
            }
        )
        self.pending_color_events.add(event)

        not_visible_match = re.fullmatch(
            r"EV_(red|green|blue)_not_visible", event
        )
        if not_visible_match is not None:
            self.reached_color_latches.discard(not_visible_match.group(1))

        match = re.fullmatch(r"EV_(red|green|blue)_reached", event)
        if match is None or self.all_colors_reached:
            return

        color = match.group(1)
        if color in self.reached_color_latches:
            return

        expected_color = self.target_color_order[self.reached_color_index]
        if color != expected_color:
            self.get_logger().debug(
                f"Reached {color} out of order; next required color is "
                f"{expected_color}. This reach does not count."
            )
            return

        self.reached_color_latches.add(color)
        self.reached_color_sequence.append(color)
        self.reached_color_index += 1
        self.task_progress_pub.publish(Int32(data=self.reached_color_index))
        self.task_progress_event_pub.publish(String(data=color))
        self.get_logger().info(
            f"Reached {color}; progress: "
            f"{self.reached_color_index}/{len(self.target_color_order)} colors"
        )

        if (
            self.reached_color_index == len(self.target_color_order)
            and not self.task_complete_sent
        ):
            self.all_colors_reached = True
            self.active_event = None
            self.motion_until = 0.0
            self.full_rotate_active = False
            self.turn_settle_until = 0.0
            self.escape_phase = None
            self.active_twist = Twist()
            self.active_twist.linear.x = self.completion_retreat_linear_x
            self.completion_parking_phase = "retreat"
            self.completion_parking_until = (
                time.time() + self.completion_retreat_duration_s
            )
            self.completion_parking_started_at = time.time()
            self._publish_cmd(self.active_twist)
            self.get_logger().info(
                "All color targets reached in order "
                f"{', '.join(self.reached_color_sequence)}; "
                "retreating from the final zone."
            )

    def _hold_after_completion(self):
        """Stop and disable the physical rover after mission completion."""
        if self.completion_shutdown_timer is not None:
            self.completion_shutdown_timer.cancel()
            self.completion_shutdown_timer = None

        # Cancel every motion mode and keep the rover stopped.
        self.active_event = None
        self.motion_until = 0.0
        self.full_rotate_active = False
        self.turn_settle_until = 0.0
        self.escape_phase = None

        self._publish_stop()
        self.enabled = False
        self._publish_mission_status()
        self.get_logger().info(
            "Patrolling task complete; supervisor disabled and rover stopped."
        )

    def _finish_completion_parking(self, reason: str):
        self.completion_parking_reason = reason
        self.completion_parking_phase = None
        if reason == "stopped in front of wall" and not self.task_complete_sent:
            self.task_complete_sent = True
            self.task_complete_pub.publish(Bool(data=True))
        self._hold_after_completion()
        self.get_logger().info(f"Completion parking finished: {reason}")

    def _parking_wall_callback(self, message: Float32):
        self.parking_wall_distance = float(message.data)
        self.parking_wall_updated_at = time.time()

    def _run_completion_parking(self, now: float):
        """Find a wall, escape blocked approaches, and stop facing the wall."""
        if now - self.completion_parking_started_at >= self.completion_parking_timeout_s:
            self._finish_completion_parking("wall parking timeout")
            return
        if self.completion_parking_phase == "escape":
            if self.escape_phase is not None:
                self._run_zone_escape(now)
                return
            self.completion_parking_phase = "turn_settle"
            self.completion_parking_until = max(now, self.turn_settle_until)

        if self.completion_parking_phase == "retreat":
            if now < self.completion_parking_until:
                self._publish_cmd(self.active_twist)
                return
            self.completion_parking_phase = "turn_left"
            self.completion_parking_until = now + math.pi / (2.0 * self.completion_parking_turn_angular_z)
            self.active_twist = Twist()
            self.active_twist.angular.z = self.completion_parking_turn_angular_z

        if self.completion_parking_phase == "turn_left":
            if now < self.completion_parking_until:
                self._publish_cmd(self.active_twist)
                return
            self.completion_parking_phase = "turn_settle"
            self.completion_parking_until = now + self.post_turn_settle_s
            self._publish_stop()
            return

        if self.completion_parking_phase == "turn_settle":
            if now < self.completion_parking_until:
                self._publish_stop()
                return
            self.completion_parking_phase = "wall_approach"

        twist = Twist()
        zones = self._effective_obstacle_zones()
        # Missing depth must never result in forward motion.
        if now - self.parking_wall_updated_at > 0.5:
            self.completion_parking_phase = "depth_wait"
            self.active_twist = twist
            self._publish_cmd(twist)
            return
        distance = self.parking_wall_distance
        wall_visible = math.isfinite(distance) and distance > 0.0
        if wall_visible and distance <= self.completion_parking_stop_distance_m:
            self._finish_completion_parking("stopped in front of wall")
            return

        if any(zone in {"LEFT", "RIGHT", "CORNER"} for zone in zones):
            self.completion_parking_phase = "obstacle_wait"
            self._publish_stop()
            if self.completion_parking_blocked_since is None:
                self.completion_parking_blocked_since = now
            if now - self.completion_parking_blocked_since >= self.motion_stuck_timeout_s:
                sign = -1.0 if "LEFT" in zones else 1.0
                self._start_livelock_escape(now, sign, "completion parking blocked")
                self.completion_parking_phase = "escape"
                self.completion_parking_blocked_since = None
                self._run_zone_escape(now)
            return

        self.completion_parking_blocked_since = None
        if not wall_visible:
            # A fresh depth frame without a broad wall calls for a new heading.
            self.completion_parking_phase = "wall_search"
            twist.angular.z = self.completion_parking_search_angular_z
        else:
            self.completion_parking_phase = "wall_approach"
            twist.linear.x = self.completion_parking_linear_x
            if self._record_motion_for_livelock(now, twist.linear.x, 0.0):
                self.completion_parking_phase = "escape"
                self._run_zone_escape(now)
                return
        self.active_twist = twist
        self._publish_cmd(twist)

    def _publish_mission_status(self):
        """Publish a latched summary for Robot Hub and other observers."""
        total = len(self.target_color_order)
        next_color = (
            self.target_color_order[self.reached_color_index]
            if self.reached_color_index < total
            else None
        )
        if self.all_colors_reached:
            state = (
                f"parking_{self.completion_parking_phase}"
                if self.enabled and self.completion_parking_phase
                else (
                    "parking_failed"
                    if self.completion_parking_reason == "wall parking timeout"
                    else "completed"
                )
            )
        elif self.enabled:
            state = "running"
        else:
            state = "stopped"
        status = json.dumps(
            {
                "mission": "patrolling",
                "completion_reason": self.completion_parking_reason,
                "state": state,
                "reached_colors": list(self.reached_color_sequence),
                "next_color": next_color,
                "progress": self.reached_color_index,
                "total": total,
                "active_event": self.active_event,
            },
            separators=(",", ":"),
        )
        if status != self._last_mission_status:
            self._last_mission_status = status
            self.mission_status_pub.publish(String(data=status))

    def _release_deferred_color_events(self):
        """Release the latest observations after the current action finishes."""
        if self._motion_event_in_progress() or not self.deferred_color_events:
            return
        events = list(self.deferred_color_events.values())
        self.deferred_color_events.clear()
        for event in events:
            self._accept_color_event(event)

    def _consume_color_event(self, event: str) -> bool:
        if event not in self.pending_color_events:
            return False
        self.pending_color_events.remove(event)
        return True

    def red_visible_check(self, sup_data):
        return self._consume_color_event("EV_red_visible")

    def red_not_visible_check(self, sup_data):
        return self._consume_color_event("EV_red_not_visible")

    def red_reached_check(self, sup_data):
        return self._consume_color_event("EV_red_reached")

    def green_visible_check(self, sup_data):
        return self._consume_color_event("EV_green_visible")

    def green_not_visible_check(self, sup_data):
        return self._consume_color_event("EV_green_not_visible")

    def green_reached_check(self, sup_data):
        return self._consume_color_event("EV_green_reached")

    def blue_visible_check(self, sup_data):
        return self._consume_color_event("EV_blue_visible")

    def blue_not_visible_check(self, sup_data):
        return self._consume_color_event("EV_blue_not_visible")

    def blue_reached_check(self, sup_data):
        return self._consume_color_event("EV_blue_reached")

    def zone_callback(self, msg: String):
        now = time.time()
        if now - self.last_zone_update < self.zone_update_min_dt:
            return
        self.last_zone_update = now

        z = msg.data.strip().upper()
        # accept only known tokens
        if z in {"LEFT", "RIGHT", "CORNER", "CLEAR"}:
            self.obstacle_zones = [z]
        else:
            self.obstacle_zones = ["CLEAR"]

        zone = self.obstacle_zones[0]
        if zone != "CLEAR":
            self.last_non_clear_obstacle_zone = zone
            self.last_non_clear_obstacle_zone_time = now
        if zone in {"LEFT", "RIGHT"} and not self.all_colors_reached:
            self._record_zone_for_livelock(now, zone)
        if zone != self.last_logged_zone:
            # self.get_logger().info(f"Detected zone changed to {zone}")
            self.last_logged_zone = zone

    # -------------------------------
    # SCT input check functions (uncontrollables)
    # -------------------------------
    def _install_uncontrollable_callbacks(self):
        # Attach callbacks only for events that exist in current supervisor YAML.
        def add(ev: str, fn):
            key = f"EV_{ev}"
            if key in self.sct.EV:
                self.sct.add_callback(self.sct.EV[key], None, fn, None)

        add("obstacle_front", self.middle_check)
        add("path_clear", self.clear_path_check)
        add("obstacle_left", self.left_check)
        add("obstacle_right", self.right_check)
        add("red_not_visible", self.red_not_visible_check)
        add("red_visible", self.red_visible_check)
        add("red_reached", self.red_reached_check)
        add("green_not_visible", self.green_not_visible_check)
        add("green_visible", self.green_visible_check)
        add("green_reached", self.green_reached_check)
        add("blue_not_visible", self.blue_not_visible_check)
        add("blue_visible", self.blue_visible_check)
        add("blue_reached", self.blue_reached_check)

    # -------------------------------
    # Motion execution
    # -------------------------------
    def _task_action_spec(self, ev_name: str) -> Optional[ActionSpec]:
        """Translate a high-level LLM request through fixed obstacle safety."""
        if ev_name in {"EV_search_color", "EV_approach_color"}:
            if self.reached_color_index >= len(self.target_color_order):
                return ActionSpec()
            target_color = self.target_color_order[self.reached_color_index]
            if ev_name == "EV_search_color":
                ev_name = f"EV_search_{target_color}"
            else:
                ev_name = f"EV_approach_{target_color}"
        if not (
            ev_name.startswith("EV_task_")
            or ev_name.startswith("EV_search_")
            or ev_name.startswith("EV_approach_")
        ):
            return None
        motion_requests = {
            "EV_task_move_forward",
            "EV_task_rotate_clockwise",
            "EV_task_rotate_counterclockwise",
            "EV_search_red",
            "EV_search_green",
            "EV_search_blue",
            "EV_approach_red",
            "EV_approach_green",
            "EV_approach_blue",
        }
        if ev_name not in motion_requests:
            return ActionSpec()
        zones = self._effective_obstacle_zones()
        if "CORNER" in zones:
            return ActionSpec(
                angular_z=self.full_rotate_omega, is_full_rotate=True
            )
        if "LEFT" in zones:
            return ActionSpec(
                angular_z=-self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        if "RIGHT" in zones:
            return ActionSpec(
                angular_z=self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        if ev_name.startswith("EV_search_"):
            search_sign = 1.0 if self.robot_index % 2 == 0 else -1.0
            return ActionSpec(
                angular_z=search_sign * self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        if ev_name.startswith("EV_approach_"):
            linear_x, angular_z = self._target_approach_components(ev_name)
            return ActionSpec(
                linear_x=linear_x,
                angular_z=angular_z,
            )
        if ev_name == "EV_task_rotate_clockwise":
            return ActionSpec(
                angular_z=-self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        if ev_name == "EV_task_rotate_counterclockwise":
            return ActionSpec(
                angular_z=self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        return ActionSpec(linear_x=self.forward_linear_x)

    def publish_twist_for_event(self, ev_name: str):
        spec = self._task_action_spec(ev_name) or self.action_table.get(ev_name)

        # Unknown controllable -> stop (safe)
        if spec is None:
            self.active_event = None
            self._publish_stop()
            return
    

        if spec.is_full_rotate:
            now = time.time()
            if (now - self.last_full_rotate_completed_at) < self.full_rotate_retrigger_block_s:
                # self.get_logger().info("full_rotate blocked by recent completion; stopping this tick")
                self.active_event = None
                self.motion_until = 0.0
                self._publish_stop()
                return
            # Cooldown: prevent repeated "scan in place" when we're stuck in obs_front
            if now < self.block_full_rotate_until:
                # self.get_logger().info("full_rotate blocked by cooldown; stopping this tick")
                self.active_event = None
                self.motion_until = 0.0
                self._publish_stop()
                return
            self.active_event = ev_name
            self._start_full_rotate(spec.angular_z)
            return

        if self._record_motion_for_livelock(
            time.time(), float(spec.linear_x), float(spec.angular_z)
        ):
            self._run_zone_escape(time.time())
            return

        # Normal pulse action
        twist = Twist()
        twist.linear.x = float(spec.linear_x)
        twist.angular.z = float(spec.angular_z)

        hold = self.motion_hold_duration if spec.hold_s is None else float(spec.hold_s)

        self.active_event = ev_name
        self.active_twist = twist
        self.motion_until = time.time() + hold
        self._publish_cmd(self.active_twist)

    # -------------------------------
    # Supervisor tick
    # -------------------------------
    def timer_callback(self):
        self._publish_mission_status()
        if not self.enabled:
            return
        now = time.time()

        # Completion has priority over contact recovery, escape and active turns.
        if self.all_colors_reached:
            self._run_completion_parking(now)
            return

        if self.escape_phase is not None:
            self._run_zone_escape(now)
            return

        # If we’re in the middle of a true full_rotate, keep executing until complete.
        if self.full_rotate_active:
            # self.get_logger().info(
            #     "FULL ROTATE active "
            #     f"(accum_rad={self.full_rotate_accum:.3f}, "
            #     f"timed_fallback={self.full_rotate_using_timed_fallback}, "
            #     f"stall_count={self.full_rotate_stall_count})"
            # )
            self._publish_cmd(self.active_twist)
            if self._update_full_rotate():
                # stop rotation and resume supervisor next tick
                self.full_rotate_active = False
                self.last_full_rotate_completed_at = now
                self._enter_post_turn_settle(now)
                # self.get_logger().info(
                #     "FULL ROTATE stopped; supervisor will select next event next tick"
                # )
            return

        if now < self.turn_settle_until:
            self._publish_stop()
            return

        # Normal pulse-hold: keep publishing until hold expires
        if self.active_event and now < self.motion_until:
            self._publish_cmd(self.active_twist)
            return

        # Otherwise: pick next event from SCT
        self.active_event = None
        self._release_deferred_color_events()
        if self.all_colors_reached:
            self._publish_stop()
            return
        self.sct.input_buffer = []
        sct_input_snapshot = self._snapshot_sct_inputs(now)
        ce_exists, ce = self.sct.run_step()
        if not ce_exists:
            # No controllable enabled -> stop
            self._log_sct_decision("none", False, sct_input_snapshot)
            self._publish_stop()
            return

        ev_name = self.ev_name_by_id.get(int(ce))
        if ev_name is None:
            self._log_sct_decision(f"unknown:{int(ce)}", True, sct_input_snapshot)
            self._publish_stop()
            return

        self._log_sct_decision(ev_name, True, sct_input_snapshot)
        # self.get_logger().info(f"Selected controllable event: {ev_name}")
        self.publish_twist_for_event(ev_name)


def main(args=None):
    rclpy.init(args=args)
    node = RobotSupervisor()
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
