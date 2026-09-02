import os
import math
import re
import time
from typing import Optional

from geometry_msgs import msg
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Int32, String
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import DeleteEntity, SetEntityPose, SpawnEntity
from tf2_msgs.msg import TFMessage

from ament_index_python.packages import get_package_share_directory
from leo_delivery.sct import SCT
from leo_supervisor_common.motion import (
    ActionSpec,
)
from leo_supervisor_common.supervisor_runtime import SupervisorRuntimeMixin
from leo_supervisor_common.target_approach import TargetApproachMixin
from leo_supervisor_common.zone_escape import ZoneLivelockEscapeMixin


TARGET_COLORS = ("red", "green", "blue")
COLOR_EVENT_RE = re.compile(
    r"EV_(?P<color>[^_]+)_(?P<target>object|zone)_"
    r"(?P<observation>not_visible|visible|reached)$"
)
DELIVERY_COMMAND_RE = re.compile(
    r"EV_(?P<action>claim)_(?P<color>[^_]+)$"
)
DELIVERY_RECEIVED_RE = re.compile(
    r"EV_received_(?P<action>claim)_(?P<color>[^_]+)$"
)


class RobotSupervisor(
    SupervisorRuntimeMixin, TargetApproachMixin, ZoneLivelockEscapeMixin, Node
):
    def __init__(self):
        super().__init__("robot_supervisor")

        # -------------------------------
        # Parameters
        # -------------------------------
        self._initialize_runtime_parameters(motion_hold_default=0.2)
        self.task_complete_sent = False
        self.robot_retired = False
        self.robot_relocation_requested = False

        # Deleting a sensor-equipped model can abort Gazebo Fortress. Retired
        # delivery robots are instead moved below the world, as in patrolling.
        self.completed_robot_z = float(
            self.declare_parameter("completed_robot_z", -100.0).value
        )
        self.set_pose_client = self.create_client(
            SetEntityPose, "/world/random_world/set_pose"
        )
        self.delete_entity_client = self.create_client(
            DeleteEntity, "/world/random_world/remove"
        )
        self.spawn_entity_client = self.create_client(
            SpawnEntity, "/world/random_world/create"
        )
        self.cargo_height = float(
            self.declare_parameter("cargo_height", 0.55).value
        )
        self.carried_box_entity: Optional[str] = None
        self.carried_box_delete_requested = False
        self.carried_box_pose_request_pending = False
        self.world_robot_pose = None
        self.pending_color_events = set()
        self.deferred_color_events = {}
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
        self.config_dir = os.path.join(get_package_share_directory("leo_delivery"), "config")
        self.explicit_yaml_path = str(self.declare_parameter("supervisor_yaml_path", "").value).strip()
        self.current_mission = "delivery"
        self._load_initial_sct()

        self._initialize_runtime_state()


        # -------------------------------
        # Publishers/Subscribers
        # -------------------------------
        self._create_runtime_interfaces()
        self.delivery_message_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.delivery_message_pub = self.create_publisher(
            String,
            "/delivery/color_messages",
            self.delivery_message_qos,
        )
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
        self.create_subscription(
            String,
            "/delivery/color_messages",
            self.delivery_message_callback,
            self.delivery_message_qos,
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
        self.sub_world_pose = self.create_subscription(
            TFMessage,
            "/world/random_world/dynamic_pose/info",
            self.world_pose_callback,
            10,
        )
        self._initialize_motion_action_table()

    def _load_sct_from_yaml(self, config_path: str):
        self.sct = SCT(
            config_path,
            random_seed=self.robot_seed,
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
            self.sct.add_callback(
                event_id,
                lambda _sup_data, action=action, color=color:
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
            if event == "EV_path_clear":
                continue
            if event.endswith(("_object_reached", "_zone_reached")):
                self.get_logger().info(f"############################# REACHED: {event} #############################")
            elif event.endswith(("_object_visible", "_zone_visible")):
                self.get_logger().info(f"############################# VISIBLE: {event} #############################")
            else:
                self.get_logger().info(f"Triggered uncontrollable event: {event}")

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
            or action not in {"claim", "delivered"}
            or color not in self.delivery_colors
        ):
            return
        sender_match = re.fullmatch(r"robot_(\d+)", sender)
        if sender_match is None:
            return
        if action == "claim" and self.claimed_delivery_color == color:
            sender_index = int(sender_match.group(1))
            if self.robot_index < sender_index:
                self.get_logger().info(
                    f"CLAIM TIE WON: keeping {color} against {sender}"
                )
                return
            self.get_logger().info(
                f"CLAIM TIE LOST: yielding {color} to {sender}"
            )
            self.claimed_delivery_color = None
            self.active_delivery_color = None
            self.delivery_phase = "object"
            self.pending_color_events.clear()
            self.deferred_color_events.clear()
        event = f"EV_received_{action}_{color}"
        if action == "delivered":
            self._delivery_message_received(color)
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
        elif action == "delivered" and self.claimed_delivery_color == color:
            self.claimed_delivery_color = None
        self.delivery_message_pub.publish(
            String(data=f"{self.ns}:{action}:{color}")
        )
        self.get_logger().info(
            f"DELIVERY PUBLISHED: {action}:{color}"
        )
        if action == "delivered":
            self._delivery_message_published(color)

    def _delivery_message_received(self, color: str):
        self._record_delivered_color(color, publish_event=False)

    def _delivery_message_published(self, color: str):
        self._record_delivered_color(color, publish_event=True)
        self._retire_robot_after_delivery()

    def _retire_robot_after_delivery(self):
        """Stop this supervisor and move its model away after one delivery."""
        if self.robot_retired:
            return
        self.robot_retired = True
        self._cancel_all_motion()
        self._publish_stop()
        self.get_logger().info(
            f"Delivery complete; retiring robot entity '{self.ns}'."
        )
        self._request_robot_relocation()

    def _request_robot_relocation(self):
        if self.robot_relocation_requested:
            return
        if not self.set_pose_client.service_is_ready():
            self.get_logger().warning(
                "Robot set-pose service is not ready; will retry."
            )
            return

        request = SetEntityPose.Request()
        request.entity.name = self.ns
        request.entity.type = Entity.MODEL
        request.pose.position.x = float(self.x)
        request.pose.position.y = float(self.y)
        request.pose.position.z = self.completed_robot_z
        request.pose.orientation.w = 1.0
        self.robot_relocation_requested = True
        future = self.set_pose_client.call_async(request)
        future.add_done_callback(self._robot_relocation_done)

    def _robot_relocation_done(self, future):
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - middleware failure
            self.robot_relocation_requested = False
            self.get_logger().error(f"Robot relocation request failed: {exc}")
            return
        if response.success:
            self.get_logger().info(
                f"Robot entity '{self.ns}' moved to "
                f"z={self.completed_robot_z:.1f}."
            )
            return
        self.robot_relocation_requested = False
        self.get_logger().error(
            f"Simulator rejected relocation of robot entity '{self.ns}'."
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
        if self.active_delivery_color is not None:
            self._pick_up_box_model(self.active_delivery_color)
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        self.delivery_phase = "zone"

    def _confirm_delivery(self, _sup_data):
        if self.carried_box_entity is not None:
            self._delete_carried_box()
        if self.active_delivery_color is not None:
            # Completion notification is runtime bookkeeping, outside the SCT
            # event alphabet.
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

    def world_pose_callback(self, message: TFMessage):
        """Capture this robot's pose in Gazebo's world coordinate frame."""
        for transform in message.transforms:
            if transform.child_frame_id != self.ns:
                continue
            self.world_robot_pose = transform.transform
            return

    def _cargo_world_pose(self):
        if self.world_robot_pose is not None:
            translation = self.world_robot_pose.translation
            rotation = self.world_robot_pose.rotation
            return (
                float(translation.x),
                float(translation.y),
                float(translation.z) + self.cargo_height,
                rotation,
            )
        # This fallback is only used during simulator startup, before the first
        # world-pose update arrives.
        orientation = msg.Quaternion()
        orientation.z = math.sin(self.yaw * 0.5)
        orientation.w = math.cos(self.yaw * 0.5)
        return float(self.x), float(self.y), self.cargo_height, orientation

    @staticmethod
    def _cargo_sdf(entity_name: str, color: str) -> str:
        rgb = {"red": "1 0 0 1", "green": "0 1 0 1", "blue": "0 0 1 1"}[color]
        return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{entity_name}">
    <static>false</static>
    <link name="cargo_link">
      <gravity>false</gravity>
      <visual name="cargo_visual">
        <geometry><box><size>0.25 0.25 0.25</size></box></geometry>
        <material><ambient>{rgb}</ambient><diffuse>{rgb}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""

    def _pick_up_box_model(self, color: str):
        """Delete the ground box and create a movable visual cargo copy."""
        if not (
            self.delete_entity_client.service_is_ready()
            and self.spawn_entity_client.service_is_ready()
        ):
            self.get_logger().error("Cargo create/remove services are not ready.")
            return

        ground_box = DeleteEntity.Request()
        ground_box.entity.name = f"{color}_delivery_box"
        ground_box.entity.type = Entity.MODEL
        self.delete_entity_client.call_async(ground_box)

        entity_name = f"{self.ns}_carried_{color}_box"
        cargo_x, cargo_y, cargo_z, cargo_orientation = self._cargo_world_pose()
        request = SpawnEntity.Request()
        request.entity_factory.name = entity_name
        request.entity_factory.allow_renaming = False
        request.entity_factory.sdf = self._cargo_sdf(entity_name, color)
        request.entity_factory.pose.position.x = cargo_x
        request.entity_factory.pose.position.y = cargo_y
        request.entity_factory.pose.position.z = cargo_z
        request.entity_factory.pose.orientation = cargo_orientation
        future = self.spawn_entity_client.call_async(request)
        future.add_done_callback(
            lambda completed, entity_name=entity_name:
                self._cargo_spawn_done(completed, entity_name)
        )

    def _cargo_spawn_done(self, future, entity_name: str):
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - middleware failure
            self.get_logger().error(f"Cargo spawn request failed: {exc}")
            return
        if not response.success:
            self.get_logger().error(f"Simulator rejected cargo spawn '{entity_name}'.")
            return
        self.carried_box_entity = entity_name
        self.carried_box_delete_requested = False
        self.get_logger().info(f"Cargo '{entity_name}' placed on robot.")
        self._update_carried_box_pose()

    def _delete_carried_box(self):
        if self.carried_box_delete_requested:
            return
        if not self.delete_entity_client.service_is_ready():
            self.get_logger().warning("Cargo remove service is not ready; will retry.")
            return
        request = DeleteEntity.Request()
        request.entity.name = self.carried_box_entity
        request.entity.type = Entity.MODEL
        self.carried_box_delete_requested = True
        future = self.delete_entity_client.call_async(request)
        future.add_done_callback(self._cargo_delete_done)

    def _cargo_delete_done(self, future):
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - middleware failure
            self.carried_box_delete_requested = False
            self.get_logger().error(f"Cargo remove request failed: {exc}")
            return
        if not response.success:
            self.carried_box_delete_requested = False
            self.get_logger().warning("Simulator rejected cargo removal.")
            return
        self.get_logger().info(f"Delivered cargo '{self.carried_box_entity}' removed.")
        self.carried_box_entity = None
        self.carried_box_delete_requested = False

    def _update_carried_box_pose(self):
        """Keep the spawned visual cargo on this robot."""
        if (
            self.carried_box_entity is None
            or self.carried_box_delete_requested
            or self.carried_box_pose_request_pending
            or not self.set_pose_client.service_is_ready()
        ):
            return
        request = SetEntityPose.Request()
        request.entity.name = self.carried_box_entity
        request.entity.type = Entity.MODEL
        cargo_x, cargo_y, cargo_z, cargo_orientation = self._cargo_world_pose()
        request.pose.position.x = cargo_x
        request.pose.position.y = cargo_y
        request.pose.position.z = cargo_z
        request.pose.orientation = cargo_orientation
        self.carried_box_pose_request_pending = True
        future = self.set_pose_client.call_async(request)
        future.add_done_callback(self._carried_box_pose_done)

    def _carried_box_pose_done(self, future):
        self.carried_box_pose_request_pending = False
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - middleware failure
            self.get_logger().error(f"Cargo pose request failed: {exc}")
            return
        if not response.success:
            self.get_logger().warning(
                f"Simulator rejected cargo pose for '{self.carried_box_entity}'."
            )
            return

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

    # -------------------------------
    # Motion execution
    # -------------------------------

    def _cancel_all_motion(self):
        self.active_event = None
        self.motion_until = 0.0
        self.active_twist = Twist()
        self.full_rotate_active = False
        self.full_rotate_accum = 0.0
        self.full_rotate_started_at = 0.0
        self.contact_recovery_until = 0.0
        self.turn_settle_until = 0.0

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
        self._update_carried_box_pose()
        if self.robot_retired:
            self._publish_stop()
            self._request_robot_relocation()
            return

        now = time.time()

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
