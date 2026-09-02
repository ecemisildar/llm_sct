import os
import re
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int32, String
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from ament_index_python.packages import get_package_share_directory
from leo_patrolling.sct import SCT
from leo_supervisor_common.motion import (
    ActionSpec,
)
from leo_supervisor_common.target_approach import TargetApproachMixin
from leo_supervisor_common.zone_escape import ZoneLivelockEscapeMixin
from leo_supervisor_common.supervisor_runtime import SupervisorRuntimeMixin


DEFAULT_TARGET_COLOR_ORDER = ("red", "green", "blue")


class RobotSupervisor(
    SupervisorRuntimeMixin, TargetApproachMixin, ZoneLivelockEscapeMixin, Node
):
    def _contact_recovery_blocked(self) -> bool:
        return self.all_colors_reached

    def __init__(self):
        super().__init__("robot_supervisor")

        # -------------------------------
        # Parameters
        # -------------------------------
        self._initialize_runtime_parameters(motion_hold_default=0.2)
        self.task_complete_sent = False

        # Runtime deletion of sensor-equipped models can crash Gazebo Fortress.
        # Completed robots are therefore stopped and moved below the world.
        self.world_name = str(
            self.declare_parameter("world_name", "random_world").value
        ).strip()
        self.robot_model_name = str(
            self.declare_parameter("robot_model_name", self.ns).value
        ).strip()
        self.remove_completed_robot = bool(
            self.declare_parameter("remove_completed_robot", True).value
        )
        self.completed_robot_z = float(
            self.declare_parameter("completed_robot_z", -100.0).value
        )
        self.set_pose_service_name = f"/world/{self.world_name}/set_pose"
        self.set_pose_client = self.create_client(
            SetEntityPose,
            self.set_pose_service_name,
        )
        self.set_pose_future = None
        self.completed_robot_moved = False
        self.pending_color_events = set()
        self.deferred_color_events = {}
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
            self.get_logger().warning(
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
        self.config_dir = os.path.join(get_package_share_directory("leo_patrolling"), "config")
        self.explicit_yaml_path = str(self.declare_parameter("supervisor_yaml_path", "").value).strip()
        self.current_mission = "patrolling"
        self._load_initial_sct()

        self._initialize_runtime_state()


        # -------------------------------
        # Publishers/Subscribers
        # -------------------------------
        self._create_runtime_interfaces()
        self.task_complete_pub = self.create_publisher(
            Bool, f"/{self.ns}/task_complete", 10
        )
        self.task_progress_pub = self.create_publisher(Int32, "task_progress", 10)
        self.task_progress_event_pub = self.create_publisher(
            String, "task_progress_event", 10
        )
        self.create_subscription(
            String,
            "color_events",
            self.color_event_callback,
            10,
        )
        self._create_target_approach_subscriptions()
        self._initialize_motion_action_table()

    def _load_sct_from_yaml(self, config_path: str):
        self.sct = SCT(
            config_path,
            random_seed=self.robot_seed,
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
        self._load_named_sct("supervisor.yaml")

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
            if event == "EV_red_reached" or event == "EV_green_reached" or event == "EV_blue_reached":
                self.get_logger().info(f"############################# REACHED: {event} #############################")
            elif event == "EV_red_visible" or event == "EV_green_visible" or event == "EV_blue_visible":
                self.get_logger().info(f"############################# VISIBLE: {event} #############################")
            # else:
            #     self.get_logger().info(f"Triggered uncontrollable event: {event}")

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
            self.get_logger().warning(
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
            self.task_complete_sent = True
            self.task_complete_pub.publish(Bool(data=True))
            self._publish_stop()
            self.get_logger().info(
                "All color targets reached in order "
                f"{', '.join(self.reached_color_sequence)}; "
                "final events published."
            )
            self._hold_after_completion()

    def _hold_after_completion(self):
        """Stop and move the completed robot outside the active Gazebo world."""
        # Cancel every motion mode before moving the model.
        self.active_event = None
        self.motion_until = 0.0
        self.full_rotate_active = False
        self.contact_recovery_until = 0.0
        self.turn_settle_until = 0.0
        self.escape_phase = None

        self._publish_stop()
        self.move_completed_robot_outside_world()

    def move_completed_robot_outside_world(self):
        """Move this robot below the world without deleting its Gazebo sensors."""
        if not self.remove_completed_robot:
            self.get_logger().info(
                "Completed robot removal is disabled; robot remains stopped."
            )
            return

        if self.completed_robot_moved:
            return

        if not self.set_pose_client.service_is_ready():
            self.get_logger().warning(
                "Gazebo set-pose service is unavailable: "
                f"{self.set_pose_service_name}. The robot will remain stopped."
            )
            return

        request = SetEntityPose.Request()
        request.entity.name = self.robot_model_name
        request.entity.type = Entity.MODEL

        request.pose.position.x = float(self.x)
        request.pose.position.y = float(self.y)
        request.pose.position.z = self.completed_robot_z
        request.pose.orientation.x = 0.0
        request.pose.orientation.y = 0.0
        request.pose.orientation.z = 0.0
        request.pose.orientation.w = 1.0

        self.completed_robot_moved = True
        self.get_logger().info(
            f"Moving completed Gazebo model '{self.robot_model_name}' "
            f"to z={self.completed_robot_z:.1f}."
        )

        self.set_pose_future = self.set_pose_client.call_async(request)
        self.set_pose_future.add_done_callback(
            self._completed_robot_pose_callback
        )

    def _completed_robot_pose_callback(self, future):
        """Handle Gazebo's response to the completed-robot pose request."""
        try:
            response = future.result()
        except Exception as exc:
            self.completed_robot_moved = False
            self.get_logger().error(
                f"Failed to move completed robot "
                f"'{self.robot_model_name}': {exc}"
            )
            return

        if response.success:
            self.get_logger().info(
                f"Completed robot '{self.robot_model_name}' "
                "was moved outside the Gazebo arena."
            )
        else:
            self.completed_robot_moved = False
            self.get_logger().error(
                f"Gazebo rejected the set-pose request for "
                f"'{self.robot_model_name}'."
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
        requested_event = ev_name
        if ev_name in {"EV_search_color", "EV_approach_color"}:
            if self.reached_color_index >= len(self.target_color_order):
                return ActionSpec()
            target_color = self.target_color_order[self.reached_color_index]
            if ev_name == "EV_search_color":
                ev_name = f"EV_search_{target_color}"
            else:
                ev_name = f"EV_approach_{target_color}"
        if requested_event not in {"EV_search_color", "EV_approach_color"}:
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

        # Completion has priority over contact recovery, escape and active turns.
        if self.all_colors_reached:
            self._publish_stop()
            return

        if now < self.contact_recovery_until:
            self._publish_contact_recovery_cmd()
            return

        if self.escape_phase is not None:
            self._run_zone_escape(now)
            return

        # If we’re in the middle of a true full_rotate, keep executing until complete.
        if self.full_rotate_active:
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
