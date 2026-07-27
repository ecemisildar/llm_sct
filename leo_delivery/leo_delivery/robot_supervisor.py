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

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int32, String
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts

from ament_index_python.packages import get_package_share_directory
from leo_delivery.sct import SCT
from leo_supervisor_common.decision_logger import SctDecisionLogger
from leo_supervisor_common.motion import (
    ActionSpec,
    wrap_to_pi as _wrap_to_pi,
    yaw_from_quat as _yaw_from_quat,
)
from leo_supervisor_common.target_approach import TargetApproachMixin
from leo_supervisor_common.zone_escape import ZoneLivelockEscapeMixin


TARGET_COLORS = ("red", "green", "blue")


class RobotSupervisor(TargetApproachMixin, ZoneLivelockEscapeMixin, Node):
    def __init__(self):
        super().__init__("robot_supervisor")

        # -------------------------------
        # Parameters
        # -------------------------------
        self.supervisor_period = float(self.declare_parameter("supervisor_period", 0.1).value)
        self.motion_hold_duration = float(self.declare_parameter("motion_hold_duration", 0.6).value)
        self.task_complete_sent = False

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
        self.contact_recovery_retrigger_block_s = float(
            self.declare_parameter("contact_recovery_retrigger_block_s", 1.0).value
        )
        self.contact_topic = str(
            self.declare_parameter(
                "contact_topic",
                f"/world/random_world/model/{self.ns}/link/{self.ns}/base_footprint/sensor/contact_sensor/contact",
            ).value
        )
        self.pending_color_events = set()
        self.deferred_color_events = {}
        self.pending_received_colors = set()
        self.reached_colors = set()
        self._initialize_target_approach()
        self.all_colors_reached = False

        # -------------------------------
        # Load SCT YAML
        # -------------------------------
        self.config_dir = os.path.join(get_package_share_directory("leo_delivery"), "config")
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
        self.contact_recovery_until = 0.0
        self.last_contact_recovery_started_at = 0.0
        self.contact_recovery_source = ""
        self.contact_recovery_turn_sign = 1.0 if self.robot_index % 2 == 0 else -1.0
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
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
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
        self.sub_odom = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.sub_contact = self.create_subscription(
            Contacts,
            self.contact_topic,
            self.contact_callback,
            10,
        )


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
            "EV_move_forward": ActionSpec(linear_x=0.3, angular_z=0.0),
            "EV_move_backward": ActionSpec(linear_x=-0.2, angular_z=0.0, hold_s=self.recovery_back_hold_s),
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
        self._install_delivery_controllable_callbacks()

    def _install_delivery_controllable_callbacks(self):
        for color in TARGET_COLORS:
            for event in (
                f"EV_pub_{color}",
                f"EV_pub_going_{color}",
                f"EV_task_pub_going_{color}",
            ):
                if event in self.sct.EV:
                    self.sct.add_callback(
                        self.sct.EV[event],
                        lambda _sup_data, color=color: self._publish_delivery_color(
                            color
                        ),
                        None,
                        None,
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
            "contact_recovery_active": now < self.contact_recovery_until,
            "turn_settle_active": now < self.turn_settle_until,
            "active_event_before": self.active_event or "",
        }

    def _log_sct_decision(self, selected_event: str, ce_exists: bool, snapshot):
        uncontrollable_events = [
            self.ev_name_by_id.get(int(event), f"unknown:{int(event)}")
            for event in self.sct.last_uncontrollable_events
        ]
        for event in uncontrollable_events:
            if event == "EV_path_clear":
                continue
            if event == "EV_red_reached" or event == "EV_green_reached" or event == "EV_blue_reached":
                self.get_logger().info(f"############################# REACHED: {event} #############################")
            elif event == "EV_red_visible" or event == "EV_green_visible" or event == "EV_blue_visible":
                self.get_logger().info(f"############################# VISIBLE: {event} #############################")
            else:
                self.get_logger().info(f"Triggered uncontrollable event: {event}")

        self.decision_logger.log(
            selected_event, ce_exists, uncontrollable_events, snapshot
        )

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

    # -------------------------------
    # Subscriptions
    # -------------------------------
    def _motion_event_in_progress(self) -> bool:
        """Return whether a selected motion event still owns the robot."""
        now = time.time()
        return (
            now < self.contact_recovery_until
            or self.full_rotate_active
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

    def delivery_message_callback(self, msg: String):
        try:
            sender, color = msg.data.strip().split(":", 1)
        except ValueError:
            return
        color = color.lower()
        if sender == self.ns or color not in TARGET_COLORS:
            return
        match = re.fullmatch(r"robot_(\d+)", sender)
        if match is None or int(match.group(1)) >= self.robot_index:
            return
        self.pending_received_colors.add(color)
        self.get_logger().info(
            f"DELIVERY RECEIVED: EV_received_{color} from {sender}"
        )
        self._cancel_all_motion()
        self._publish_stop()

    def _publish_delivery_color(self, color: str):
        self.delivery_message_pub.publish(String(data=f"{self.ns}:{color}"))
        self.get_logger().info(f"DELIVERY PUBLISHED: EV_pub_{color}")

    def _consume_received_color(self, color: str) -> bool:
        if color not in self.pending_received_colors:
            return False
        self.pending_received_colors.remove(color)
        return True

    def received_red_check(self, _sup_data):
        return self._consume_received_color("red")

    def received_green_check(self, _sup_data):
        return self._consume_received_color("green")

    def received_blue_check(self, _sup_data):
        return self._consume_received_color("blue")

    def _accept_color_event(self, event: str):
        """Queue a color event when no motion event is executing."""
        color = event.split("_")[1]
        self.pending_color_events.difference_update(
            {
                f"EV_{color}_not_visible",
                f"EV_{color}_visible",
                f"EV_{color}_reached",
            }
        )
        self.pending_color_events.add(event)

        match = re.fullmatch(r"EV_(red|green|blue)_reached", event)
        if match is None or self.all_colors_reached:
            return

        color = match.group(1)
        if color in self.reached_colors:
            return

        self.reached_colors.add(color)
        self.task_progress_pub.publish(Int32(data=len(self.reached_colors)))
        self.task_progress_event_pub.publish(String(data=color))
        self.get_logger().info(
            f"Reached {color}; progress: "
            f"{len(self.reached_colors)}/{len(TARGET_COLORS)} colors"
        )

        if not self.task_complete_sent:
            self.all_colors_reached = True
            self.task_complete_sent = True
            self.task_complete_pub.publish(Bool(data=True))
            self._publish_stop()
            self.get_logger().info(
                f"Delivery goal {color} reached; waiting for all robots."
            )

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

    def contact_callback(self, msg: Contacts):
        if not self.contact_recovery_enabled:
            return
        if not msg.contacts:
            return

        now = time.time()
        if (now - self.last_contact_recovery_started_at) < self.contact_recovery_retrigger_block_s:
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
            f"(source={self.contact_recovery_source}, duration_s={self.contact_recovery_duration_s:.2f})"
        )

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
        add("red_not_visible", self.red_not_visible_check)
        add("red_visible", self.red_visible_check)
        add("red_reached", self.red_reached_check)
        add("green_not_visible", self.green_not_visible_check)
        add("green_visible", self.green_visible_check)
        add("green_reached", self.green_reached_check)
        add("blue_not_visible", self.blue_not_visible_check)
        add("blue_visible", self.blue_visible_check)
        add("blue_reached", self.blue_reached_check)
        add("received_red", self.received_red_check)
        add("received_green", self.received_green_check)
        add("received_blue", self.received_blue_check)

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

    def _cancel_all_motion(self):
        self.active_event = None
        self.motion_until = 0.0
        self.active_twist = Twist()
        self.full_rotate_active = False
        self.full_rotate_accum = 0.0
        self.full_rotate_started_at = 0.0
        self.contact_recovery_until = 0.0
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
        """Translate a high-level LLM request through fixed obstacle safety."""
        if not ev_name.startswith("EV_task_"):
            return None
        motion_requests = {
            "EV_task_move_forward",
            "EV_task_move_backward",
            "EV_task_rotate_clockwise",
            "EV_task_rotate_counterclockwise",
            "EV_task_search_red",
            "EV_task_search_green",
            "EV_task_search_blue",
            "EV_task_approach_red",
            "EV_task_approach_green",
            "EV_task_approach_blue",
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
        if ev_name.startswith("EV_task_search_"):
            search_sign = 1.0 if self.robot_index % 2 == 0 else -1.0
            return ActionSpec(
                angular_z=search_sign * self.short_rotation_omega,
                hold_s=self.supervisor_period,
            )
        if ev_name.startswith("EV_task_approach_"):
            linear_x, angular_z = self._target_approach_components(ev_name)
            return ActionSpec(
                linear_x=linear_x,
                angular_z=angular_z,
            )
        if ev_name == "EV_task_move_backward":
            if "BACK" in zones:
                return ActionSpec()
            return ActionSpec(
                linear_x=-0.2, hold_s=self.recovery_back_hold_s
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
        return ActionSpec(linear_x=0.3)

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
        now = time.time()

        if now < self.contact_recovery_until:
            self._publish_contact_recovery_cmd()
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
