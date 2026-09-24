import os
import random
import math
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Int32, String
from std_srvs.srv import SetBool
from nav_msgs.msg import Odometry

from ament_index_python.packages import get_package_share_directory
from leo_real_experiments.delivery_sct import SCT
from leo_real_experiments.decision_logger import SctDecisionLogger
from leo_real_experiments.motion import (
    ActionSpec,
    wrap_to_pi as _wrap_to_pi,
    yaw_from_quat as _yaw_from_quat,
)
from leo_real_experiments.target_approach import TargetApproachMixin
from leo_real_experiments.zone_escape import ZoneLivelockEscapeMixin


TARGET_COLORS = ("red", "green", "blue")
COLOR_EVENT_RE = re.compile(
    r"EV_(?P<color>[^_]+)_(?P<target>object|zone)_"
    r"(?P<observation>not_visible|visible|reached)$"
)
DELIVERY_COMMAND_RE = re.compile(
    r"EV_(?P<action>claim|release|publish_delivered)_(?P<color>[^_]+)$"
)
DELIVERY_RECEIVED_RE = re.compile(
    r"EV_received_(?P<action>claim|release|delivered)_(?P<color>[^_]+)$"
)


class RobotSupervisor(TargetApproachMixin, ZoneLivelockEscapeMixin, Node):
    def __init__(self):
        super().__init__("robot_supervisor")
        self.get_logger().set_level(LoggingSeverity.WARN)

        # -------------------------------
        # Parameters
        # -------------------------------
        self.supervisor_period = float(self.declare_parameter("supervisor_period", 0.1).value)
        self.motion_hold_duration = float(self.declare_parameter("motion_hold_duration", 0.6).value)
        self.enabled = bool(self.declare_parameter("enabled", True).value)
        self.enable_service = self.create_service(
            SetBool, "/enable_supervisor", self._set_enabled
        )
        self.task_complete_sent = False
        self.robot_retired = False

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
        default_forward_probability = 0.35
        self.forward_probability = float(
            self.declare_parameter(
                "forward_probability",
                default_forward_probability,
            ).value
        )
        self.pending_color_events = set()
        self.deferred_color_events = {}
        self.last_logged_color_events = {}
        self.pending_delivery_events = set()
        self.delivered_colors = set()
        self.delivery_colors = set(TARGET_COLORS)
        self.delivery_phase = "object"
        self.claimed_delivery_color: Optional[str] = None
        self.active_delivery_color: Optional[str] = None
        self._initialize_target_approach()
        self.delivery_target_offsets = {
            (color, target): float("nan")
            for color in TARGET_COLORS
            for target in ("object", "zone")
        }
        self.all_colors_reached = False

        # -------------------------------
        # Load SCT YAML
        # -------------------------------
        self.config_dir = os.path.join(get_package_share_directory("leo_real_experiments"), "config")
        self.explicit_yaml_path = str(self.declare_parameter("supervisor_yaml_path", "").value).strip()
        self.current_mission = "delivery"
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
        self.delivery_message_pub = self.create_publisher(
            String, "/delivery/color_messages", 10
        )
        self.task_complete_pub = self.create_publisher(
            Bool, f"/{self.ns}/task_complete", 10
        )
        self.task_progress_pub = self.create_publisher(Int32, "task_progress", 10)
        self.task_progress_event_pub = self.create_publisher(
            String, "task_progress_event", 10
        )
        self.sub_zone = self.create_subscription(String, "detected_zones", self.zone_callback, 10)
        self.create_subscription(
            String,
            "color_events",
            self.color_event_callback,
            10,
        )
        self.create_subscription(
            String,
            "/delivery/color_messages",
            self.delivery_message_callback,
            10,
        )
        self._create_target_approach_subscriptions()
        self.delivery_target_offset_subscriptions = [
            self.create_subscription(
                Float32,
                f"{color}_{target}_horizontal_offset",
                lambda message, color=color, target=target:
                    self._delivery_target_offset_callback(color, target, message),
                10,
            )
            for color in TARGET_COLORS
            for target in ("object", "zone")
        ]
        self._create_odom_subscriptions()


        # -------------------------------
        # SCT callbacks for UCEs (data-driven, but still attaches known sensors)
        # -------------------------------
        # Supervisor timer
        self.timer = self.create_timer(self.supervisor_period, self.timer_callback)

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
        yaml_colors = {
            match.group("color")
            for event in self.sct.EV
            for pattern in (COLOR_EVENT_RE, DELIVERY_COMMAND_RE, DELIVERY_RECEIVED_RE)
            if (match := pattern.fullmatch(event)) is not None
        }
        if yaml_colors:
            self.delivery_colors = yaml_colors
        for ev_id in self.sct.EV.values():
            if ev_id not in self.sct.callback:
                self.sct.callback[ev_id] = {
                    "callback": None,
                    "check_input": (lambda _sup_data: False),
                    "sup_data": None,
                }
        self._install_uncontrollable_callbacks()
        self._install_delivery_controllable_callbacks()

    def _install_delivery_controllable_callbacks(self):
        for event, event_id in self.sct.EV.items():
            match = DELIVERY_COMMAND_RE.fullmatch(event)
            if match is None:
                continue
            action = match.group("action")
            color = match.group("color")
            message_action = "delivered" if action == "publish_delivered" else action
            self.sct.add_callback(
                event_id,
                lambda _sup_data, action=message_action, color=color:
                    self._publish_delivery_message(action, color),
                None,
                None,
            )
        for event, callback in (
            ("EV_pick_up_object", self._confirm_pickup),
            ("EV_drop_object", self._confirm_delivery),
        ):
            if event in self.sct.EV:
                self.sct.add_callback(
                    self.sct.EV[event], callback, None, None
                )

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
        config_path = os.path.join(self.config_dir, "sup_delivery.yaml")
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

    def _publish_cmd(self, twist: Twist):
        self.executed_event_pub.publish(String(data=self._current_command_event_label()))
        self.cmd_pub.publish(twist)

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
                r"EV_(red|green|blue)_(object|zone)_"
                r"(not_visible|visible|reached)",
                event,
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

    def destroy_node(self):
        self.decision_logger.write_summary()
        super().destroy_node()

    def _current_command_event_label(self) -> str:
        if self.escape_phase is not None:
            return f"ZONE_ESCAPE_{self.escape_phase.upper()}"
        if self.full_rotate_active:
            return self.active_event or "FULL_ROTATE"
        return self.active_event or "none"

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
        source_event = msg.data.strip()
        delivery_match = COLOR_EVENT_RE.fullmatch(source_event)
        if delivery_match is not None:
            if source_event not in self.sct.EV:
                return
            color = delivery_match.group("color")
            target = delivery_match.group("target")
            # The delivery detector publishes object and zone observations in
            # parallel. Only feed the target type for the current task phase to
            # the supervisor: object before pickup, zone after pickup.
            if target != self.delivery_phase:
                return
            if (
                self.claimed_delivery_color is not None
                and color != self.claimed_delivery_color
            ):
                return
            event = source_event
            if self._motion_event_in_progress():
                self.deferred_color_events[
                    f"{color}_{target}"
                ] = event
                return
            self._accept_color_event(event)
            return
        match = re.fullmatch(
            r"EV_(red|green|blue)_(not_visible|visible|reached)",
            source_event,
        )
        if match is None:
            return

        color, observation = match.groups()
        if (
            self.claimed_delivery_color is not None
            and color != self.claimed_delivery_color
        ):
            return
        event = f"EV_{color}_{self.delivery_phase}_{observation}"
        if event not in self.sct.EV:
            return
        if self._motion_event_in_progress():
            # Retain only the newest observation for this color. It will not be
            # presented to the SCT until the executing action has finished.
            self.deferred_color_events[color] = event
            return

        self._accept_color_event(event)

    def delivery_message_callback(self, msg: String):
        try:
            sender, action, color = msg.data.strip().split(":", 2)
        except ValueError:
            return
        action = action.lower()
        color = color.lower()
        if (
            sender == self.ns
            or action not in {"claim", "release", "delivered"}
            or color not in self.delivery_colors
        ):
            return
        if re.fullmatch(r"robot_(\d+)", sender) is None:
            return
        event = f"EV_received_{action}_{color}"
        if action == "delivered":
            self._record_delivered_color(color, publish_event=False)
        if event not in self.sct.EV:
            return
        self.pending_delivery_events.add(event)
        self.get_logger().info(
            f"DELIVERY RECEIVED: {event} from {sender}"
        )
        self._cancel_all_motion()
        self._publish_stop()

    def _publish_delivery_message(self, action: str, color: str):
        if action == "claim":
            self.claimed_delivery_color = color
            self.active_delivery_color = color
            self._discard_other_color_observations(color)
        elif (
            action in {"release", "delivered"}
            and self.claimed_delivery_color == color
        ):
            self.claimed_delivery_color = None
        self.delivery_message_pub.publish(
            String(data=f"{self.ns}:{action}:{color}")
        )
        self.get_logger().info(
            f"DELIVERY PUBLISHED: {action}:{color}"
        )
        if action == "delivered":
            self._record_delivered_color(color, publish_event=True)
            self._retire_robot_after_delivery()

    def _retire_robot_after_delivery(self):
        """Stop this physical rover after one delivery."""
        if self.robot_retired:
            return
        self.robot_retired = True
        self._cancel_all_motion()
        self._publish_stop()
        self.get_logger().info(
            f"Delivery complete; retiring robot entity '{self.ns}'."
        )

    def _record_delivered_color(self, color: str, publish_event: bool):
        if color in self.delivered_colors:
            return
        self.delivered_colors.add(color)
        self.task_progress_pub.publish(Int32(data=len(self.delivered_colors)))
        if publish_event:
            self.task_progress_event_pub.publish(String(data=color))
        if (
            self.delivery_colors
            and self.delivered_colors >= self.delivery_colors
            and not self.task_complete_sent
        ):
            self.all_colors_reached = True
            self.task_complete_sent = True
            self.task_complete_pub.publish(Bool(data=True))
            self.get_logger().info("All delivery colors completed by the team.")

    def _consume_delivery_event(self, event: str) -> bool:
        if event not in self.pending_delivery_events:
            return False
        self.pending_delivery_events.remove(event)
        return True

    def _confirm_pickup(self, _sup_data):
        if "EV_pickup_confirmed" in self.sct.EV:
            self.pending_delivery_events.add("EV_pickup_confirmed")
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        self.delivery_phase = "zone"

    def _confirm_delivery(self, _sup_data):
        if "EV_delivery_confirmed" in self.sct.EV:
            self.pending_delivery_events.add("EV_delivery_confirmed")
        if self.active_delivery_color is not None:
            # The 38-event baseline YAML has no publish-delivered controllable
            # event. Treat EV_drop_object itself as successful completion and
            # notify the rest of the team out of band.
            self._publish_delivery_message(
                "delivered", self.active_delivery_color
            )
            self.active_delivery_color = None
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        self.delivery_phase = "object"

    def _discard_other_color_observations(self, claimed_color: str):
        """Keep only observations belonging to this robot's claimed color."""
        self.pending_color_events = {
            event
            for event in self.pending_color_events
            if event.startswith(f"EV_{claimed_color}_")
        }
        self.deferred_color_events = {
            key: event
            for key, event in self.deferred_color_events.items()
            if event.startswith(f"EV_{claimed_color}_")
        }

    def _delivery_target_offset_callback(
        self, color: str, target: str, message: Float32
    ):
        self.delivery_target_offsets[(color, target)] = float(message.data)

    def _delivery_target_approach_components(self, target: str):
        color = self.claimed_delivery_color or self.active_delivery_color
        offset = self.delivery_target_offsets.get(
            (color, target), float("nan")
        )
        angular_z = (
            -self.approach_steering_gain * offset
            if math.isfinite(offset)
            else 0.0
        )
        return self.approach_linear_x, angular_z

    def _accept_color_event(self, event: str):
        """Queue a color event when no motion event is executing."""
        color = event.split("_")[1]
        color_key = f"{color}_{self.delivery_phase}"
        if self.last_logged_color_events.get(color_key) != event:
            self.last_logged_color_events[color_key] = event
            self.get_logger().warning(
                f"Triggered uncontrollable event: {event}"
            )
        self.pending_color_events.difference_update(
            {
                f"EV_{color}_{self.delivery_phase}_not_visible",
                f"EV_{color}_{self.delivery_phase}_visible",
                f"EV_{color}_{self.delivery_phase}_reached",
            }
        )
        self.pending_color_events.add(event)

        reached = COLOR_EVENT_RE.fullmatch(event)
        if reached is not None and self.claimed_delivery_color is None:
            self.active_delivery_color = reached.group("color")

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

    def odom_callback(self, msg: Odometry):
        self.have_odom = True
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self.odom_stamp_sec = int(msg.header.stamp.sec)
        self.odom_stamp_nsec = int(msg.header.stamp.nanosec)

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
        if zone in {"LEFT", "RIGHT"}:
            self._record_zone_for_livelock(now, zone)
        if zone != self.last_logged_zone:
            # self.get_logger().info(f"Detected zone changed to {zone}")
            self.last_logged_zone = zone

    def _effective_obstacle_zones(self):
        zones = []
        if self.obstacle_zones[0] != "CLEAR":
            zones.extend(self.obstacle_zones)
        elif (
            self.last_non_clear_obstacle_zone != "CLEAR"
            and (time.time() - self.last_non_clear_obstacle_zone_time) <= self.obstacle_zone_memory_s
        ):
            zones.append(self.last_non_clear_obstacle_zone)


        if not zones:
            return ["CLEAR"]

        unique_zones = []
        for zone in zones:
            if zone not in unique_zones:
                unique_zones.append(zone)
        return unique_zones

    # -------------------------------
    # SCT input check functions (uncontrollables)
    # -------------------------------
    def clear_path_check(self, sup_data):
        depth_obstacles = {"LEFT", "RIGHT", "CORNER"}
        return not any(zone in depth_obstacles for zone in self._effective_obstacle_zones())

    def middle_check(self, sup_data):
        hit = "CORNER" in self._effective_obstacle_zones()
        if hit and not self.front_obstacle_active:
            now = time.time()
            # If we're in a front-obstacle situation, block full_rotate for ~1 tick
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
        for event in self.sct.EV:
            if COLOR_EVENT_RE.fullmatch(event):
                add(
                    event.removeprefix("EV_"),
                    lambda _sup_data, event=event: self._consume_color_event(event),
                )
            elif DELIVERY_RECEIVED_RE.fullmatch(event):
                add(
                    event.removeprefix("EV_"),
                    lambda _sup_data, event=event: self._consume_delivery_event(event),
                )
        for event in (
            "pickup_confirmed",
            "pickup_failed",
            "delivery_confirmed",
            "delivery_failed",
        ):
            add(
                event,
                lambda _sup_data, event=event: self._consume_delivery_event(
                    f"EV_{event}"
                ),
            )

    def _namespace_index(self) -> int:
        if self.ns.startswith("robot_"):
            try:
                return int(self.ns.split("_")[-1])
            except ValueError:
                return 0
        return 0

    # -------------------------------
    # Motion execution
    # -------------------------------
    def _publish_stop(self):
        self.active_twist = Twist()
        self._publish_cmd(self.active_twist)

    def _cancel_all_motion(self):
        self.active_event = None
        self.motion_until = 0.0
        self.active_twist = Twist()
        self.full_rotate_active = False
        self.full_rotate_accum = 0.0
        self.full_rotate_started_at = 0.0
        self.turn_settle_until = 0.0

    def _enter_post_turn_settle(self, now: float):
        # Give sensing callbacks a short window to catch up before asking SCT
        # for another controllable event. This prevents turn retriggers when the
        # supervisor period is faster than obstacle updates.
        self.turn_settle_until = max(self.turn_settle_until, now + self.post_turn_settle_s)
        self.active_event = None
        self.motion_until = 0.0
        self._publish_stop()

    def _start_full_rotate(self, omega: float):
        # Start a 180° rotation that persists independently of the normal motion hold.
        self.full_rotate_active = True
        self.full_rotate_started_at = time.time()
        self.full_rotate_accum = 0.0
        self.full_rotate_using_timed_fallback = False
        self.full_rotate_stall_count = 0

        if self.have_odom:
            self.full_rotate_prev_yaw = self.yaw
        else:
            # fallback: timed rotation
            # duration = target / omega
            dur = abs(self.full_rotate_target_rad / max(1e-6, abs(omega)))
            self.motion_until = time.time() + min(dur, self.full_rotate_timeout_s)

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = omega
        self.active_twist = twist
        self.get_logger().info(
            "Starting FULL ROTATE "
            f"(omega={omega:.3f}, have_odom={self.have_odom}, "
            f"target_rad={self.full_rotate_target_rad:.3f}, timeout_s={self.full_rotate_timeout_s:.3f})"
        )
        self._publish_cmd(self.active_twist)

    def _update_full_rotate(self) -> bool:
        """
        Returns True if full rotate completed, False otherwise.
        """
        now = time.time()
        if (now - self.full_rotate_started_at) > self.full_rotate_timeout_s:
            # timeout safety
            self.get_logger().info("FULL ROTATE completed by timeout")
            return True

        if not self.have_odom or self.full_rotate_using_timed_fallback:
            # timed fallback uses motion_until
            done = now >= self.motion_until
            if done:
                self.get_logger().info("FULL ROTATE completed by timed fallback")
            return done

        # integrate yaw delta each tick (and keep publishing until done)
        dy = _wrap_to_pi(self.yaw - self.full_rotate_prev_yaw)
        self.full_rotate_accum += abs(dy)
        self.full_rotate_prev_yaw = self.yaw

        if abs(dy) < 1e-3:
            self.full_rotate_stall_count += 1
        else:
            self.full_rotate_stall_count = 0

        if self.full_rotate_stall_count >= 5:
            remaining = max(0.0, self.full_rotate_target_rad - self.full_rotate_accum)
            omega = max(1e-6, abs(self.active_twist.angular.z))
            dur = min(remaining / omega, self.full_rotate_timeout_s)
            self.motion_until = now + dur
            self.full_rotate_using_timed_fallback = True
            self.get_logger().info(
                "FULL ROTATE switching to timed fallback "
                f"(accum_rad={self.full_rotate_accum:.3f}, remaining_rad={remaining:.3f}, "
                f"stall_count={self.full_rotate_stall_count}, fallback_s={dur:.3f})"
            )

        done = self.full_rotate_accum >= self.full_rotate_target_rad
        if done:
            self.get_logger().info(
                f"FULL ROTATE completed by odom (accum_rad={self.full_rotate_accum:.3f})"
            )
        return done

    def _task_action_spec(self, ev_name: str) -> Optional[ActionSpec]:
        """Translate delivery requests through fixed obstacle safety."""
        delivery_motion = {
            "EV_search_object",
            "EV_approach_object",
            "EV_search_zone",
            "EV_approach_zone",
        }
        delivery_actions = {
            "EV_pick_up_object",
            "EV_drop_object",
        }
        if ev_name in delivery_actions or DELIVERY_COMMAND_RE.fullmatch(ev_name):
            return ActionSpec()
        if ev_name not in delivery_motion:
            return None
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
        if ev_name in {"EV_search_object", "EV_search_zone"}:
            search_sign = 1.0 if self.robot_index % 2 == 0 else -1.0
            return ActionSpec(
                angular_z=search_sign * self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        if ev_name in {"EV_approach_object", "EV_approach_zone"}:
            target = ev_name.removeprefix("EV_approach_")
            linear_x, angular_z = self._delivery_target_approach_components(
                target
            )
            return ActionSpec(
                linear_x=linear_x,
                angular_z=angular_z,
            )
        return ActionSpec()

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
    def _set_enabled(self, request, response):
        was_enabled = self.enabled
        self.enabled = bool(request.data)
        if was_enabled and not self.enabled:
            self._cancel_all_motion()
            self._publish_stop()
        response.success = True
        response.message = "supervisor enabled" if self.enabled else "supervisor disabled"
        return response

    def timer_callback(self):
        if not self.enabled:
            return
        if self.robot_retired:
            self._publish_stop()
            return

        now = time.time()

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
        self.get_logger().info(f"Selected controllable event: {ev_name}")
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
