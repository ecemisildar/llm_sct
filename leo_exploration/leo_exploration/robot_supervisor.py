import os
import random
import math
import re
import time
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Contacts

from ament_index_python.packages import get_package_share_directory
from leo_exploration.sct import SCT
from leo_supervisor_common.decision_logger import SctDecisionLogger
from leo_supervisor_common.motion import (
    ActionSpec,
    wrap_to_pi as _wrap_to_pi,
    yaw_from_quat as _yaw_from_quat,
)
from leo_supervisor_common.target_approach import TargetApproachMixin
from leo_supervisor_common.zone_escape import ZoneLivelockEscapeMixin
from leo_supervisor_common.supervisor_runtime import SupervisorRuntimeMixin


class RobotSupervisor(
    SupervisorRuntimeMixin, TargetApproachMixin, ZoneLivelockEscapeMixin, Node
):
    def __init__(self):
        super().__init__("robot_supervisor")

        # -------------------------------
        # Parameters
        # -------------------------------
        self.supervisor_period = float(self.declare_parameter("supervisor_period", 0.1).value)
        self.motion_hold_duration = float(self.declare_parameter("motion_hold_duration", 0.2).value)
        self._initialize_target_approach()

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
        default_forward_probability = 0.50 if self.total_robots >= 10 else 0.90
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

        # -------------------------------
        # Load SCT YAML
        # -------------------------------
        self.config_dir = os.path.join(get_package_share_directory("leo_exploration"), "config")
        self.explicit_yaml_path = str(self.declare_parameter("supervisor_yaml_path", "").value).strip()
        self.current_mission = "explore"
        self.current_yaml_path = ""
        self._load_initial_sct()
        self._last_printed_sup_states: Optional[Tuple[int, ...]] = None

        # -------------------------------
        # State (sensing)
        # -------------------------------
        self.obstacle_zones = ["CLEAR"]
        self.last_non_clear_obstacle_zone = "CLEAR"
        self.last_non_clear_obstacle_zone_time = 0.0
        self.front_obstacle_distance_m = float("inf")
        self.left_obstacle_distance_m = float("inf")
        self.right_obstacle_distance_m = float("inf")
        self.last_front_obstacle_distance_time = 0.0
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
            snapshot_columns=(
                "front_distance_m",
                "left_distance_m",
                "right_distance_m",
            ),
        )


        # -------------------------------
        # Publishers/Subscribers
        # -------------------------------
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.executed_event_pub = self.create_publisher(String, "executed_event", 10)

        self.sub_zone = self.create_subscription(String, "detected_zones", self.zone_callback, 10)
        self._create_target_approach_subscriptions()
        self.sub_front_obstacle_distance = self.create_subscription(
            Float32,
            "front_obstacle_distance",
            self.front_obstacle_distance_callback,
            10,
        )
        self.sub_left_obstacle_distance = self.create_subscription(
            Float32,
            "left_obstacle_distance",
            self.left_obstacle_distance_callback,
            10,
        )
        self.sub_right_obstacle_distance = self.create_subscription(
            Float32,
            "right_obstacle_distance",
            self.right_obstacle_distance_callback,
            10,
        )
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

    def _canonical_mission_name(self, mission: str) -> str:
        key = str(mission or "").strip().lower().replace("-", "_").replace(" ", "_")
        if key != "explore":
            key = "explore"
        return key

    def _mission_yaml_candidates(self, mission: str):
        mission_key = self._canonical_mission_name(mission)
        candidates = []
        preferred = os.path.join(self.config_dir, f"{mission_key}_sup_gpt.yaml")
        if os.path.exists(preferred):
            candidates.append(preferred)
        task_files = sorted(
            glob(os.path.join(self.config_dir, f"{mission_key}_sup_gpt_*.yaml")),
            key=os.path.getmtime,
            reverse=True,
        )
        candidates.extend(task_files)
        fallback = os.path.join(self.config_dir, "sup_gpt.yaml")
        if os.path.exists(fallback):
            candidates.append(fallback)

        seen = set()
        unique = []
        for path in candidates:
            if path not in seen:
                unique.append(path)
                seen.add(path)
        return unique

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
        paths = self._mission_yaml_candidates(self.current_mission)
        if not paths:
            self.get_logger().error(
                f"No supervisor YAML found in {self.config_dir}. "
                "Expected mission-specific files or sup_gpt.yaml."
            )
            raise SystemExit(1)
        config_path = paths[0]
        self._load_sct_from_yaml(config_path)
        self.current_yaml_path = config_path
        self.get_logger().info(
            f"Loaded initial mission '{self.current_mission}' from {os.path.basename(config_path)}"
        )

    def _print_current_state(self):
        states = tuple(int(s) for s in self.sct.sup_current_state)
        if states == self._last_printed_sup_states:
            return
        self._last_printed_sup_states = states
        # print(
        #     f"[robot_supervisor] mission={self.current_mission} current_state={states}",
        #     flush=True,
        # )

    def _snapshot_sct_inputs(self, now: float):
        effective_zones = self._effective_obstacle_zones()
        depth_obstacles = {"LEFT", "RIGHT", "CORNER"}
        return {
            "elapsed_s": now - self.supervisor_started_at,
            "raw_zone": self.obstacle_zones[0],
            "effective_zone": "|".join(effective_zones),
            "front_distance_m": self.front_obstacle_distance_m,
            "left_distance_m": self.left_obstacle_distance_m,
            "right_distance_m": self.right_obstacle_distance_m,
            "distance_age_s": now - self.last_front_obstacle_distance_time,
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
        self.decision_logger.log(
            selected_event, ce_exists, uncontrollable_events, snapshot
        )

    # -------------------------------
    # Subscriptions
    # -------------------------------
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
            self.get_logger().info(f"Detected zone changed to {zone}")
            self.last_logged_zone = zone

    def front_obstacle_distance_callback(self, msg: Float32):
        distance = float(msg.data)
        self.front_obstacle_distance_m = distance if math.isfinite(distance) else float("inf")
        self.last_front_obstacle_distance_time = time.time()

    def left_obstacle_distance_callback(self, msg: Float32):
        distance = float(msg.data)
        self.left_obstacle_distance_m = distance if math.isfinite(distance) else float("inf")

    def right_obstacle_distance_callback(self, msg: Float32):
        distance = float(msg.data)
        self.right_obstacle_distance_m = distance if math.isfinite(distance) else float("inf")

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

    # -------------------------------
    # Motion execution
    # -------------------------------
    def _task_action_spec(self, ev_name: str) -> Optional[ActionSpec]:
        """Translate a high-level LLM request through fixed obstacle safety."""
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
            return ActionSpec(linear_x=linear_x, angular_z=angular_z)
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
                self.get_logger().info("full_rotate blocked by recent completion; stopping this tick")
                self.active_event = None
                self.motion_until = 0.0
                self._publish_stop()
                return
            # Cooldown: prevent repeated "scan in place" when we're stuck in obs_front
            if now < self.block_full_rotate_until:
                self.get_logger().info("full_rotate blocked by cooldown; stopping this tick")
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
            self.get_logger().info(
                "FULL ROTATE active "
                f"(accum_rad={self.full_rotate_accum:.3f}, "
                f"timed_fallback={self.full_rotate_using_timed_fallback}, "
                f"stall_count={self.full_rotate_stall_count})"
            )
            self._publish_cmd(self.active_twist)
            if self._update_full_rotate():
                # stop rotation and resume supervisor next tick
                self.full_rotate_active = False
                self.last_full_rotate_completed_at = now
                self._enter_post_turn_settle(now)
                self.get_logger().info(
                    "FULL ROTATE stopped; supervisor will select next event next tick"
                )
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
        self.sct.input_buffer = []
        sct_input_snapshot = self._snapshot_sct_inputs(now)
        ce_exists, ce = self.sct.run_step()
        self._print_current_state()
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
