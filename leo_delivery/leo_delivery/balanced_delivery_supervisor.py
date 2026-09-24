"""Generic six-box delivery balanced between red and blue zones."""

import math
import re
import xml.etree.ElementTree as ET

import rclpy
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import Bool, Int32, String

from leo_delivery.robot_supervisor import RobotSupervisor


SOURCE_EVENT_RE = re.compile(
    r"EV_(?P<color>red|green|blue)_(?P<kind>object|zone)_"
    r"(?P<observation>not_visible|visible|reached)$"
)
COLOR_EVENT_RE = re.compile(
    r"EV_(?P<color>red|green|blue)_"
    r"(?P<observation>not_visible|visible|reached)$"
)
MESSAGE_RE = re.compile(r"(?P<sender>robot_\d+):delivered:(?P<zone>zone_[ab])$")
TOTAL_DELIVERY_BOXES = 6
STACK_EVENT_ALIASES = {
    **{
        f"EV_zone_{zone}_{observation}": f"EV_{color}_{observation}"
        for zone, color in (("a", "red"), ("b", "blue"))
        for observation in ("not_visible", "visible", "reached", "not_reached")
    },
    "EV_drop_zone_a": "EV_drop_zone_red",
    "EV_drop_zone_b": "EV_drop_zone_blue",
    "EV_received_delivered_zone_a": "EV_recieve_drop_zone_red",
    "EV_received_delivered_zone_b": "EV_recieve_drop_zone_blue",
}


class BalancedDeliverySupervisor(RobotSupervisor):
    """Map colored camera observations onto the generic YAML event alphabet."""

    def __init__(self):
        self.balance_state = "equal"
        self.generic_phase = "object"
        self.active_object_color = "green"
        self.active_zone_color = "red"
        self.box_consumed = False
        self.local_delivery_count = 0
        self.team_delivery_count = 0
        self.pickup_boxes = {}
        self.delivery_zones = {}
        self.consumed_boxes = set()
        self.selected_delivery_zone = None
        super().__init__()
        world_path = str(self.declare_parameter("pickup_world_path", "").value)
        if world_path:
            for model in ET.parse(world_path).findall(".//world/model"):
                name = model.get("name", "")
                if name in {"red_box", "blue_box"}:
                    pose = [float(v) for v in model.findtext("pose").split()]
                    zone = "zone_a" if name == "red_box" else "zone_b"
                    self.delivery_zones[zone] = (pose[0], pose[1])
                    continue
                if not name.startswith("generic_delivery_box_"):
                    continue
                pose = [float(v) for v in model.findtext("pose").split()]
                size = [float(v) for v in model.findtext(".//collision/geometry/box/size").split()]
                self.pickup_boxes[name] = (pose[0], pose[1], size[0] / 2, size[1] / 2)
        self.balance_message_pub = self.create_publisher(
            String, "/delivery/balanced_messages", self.delivery_message_qos
        )
        self.create_subscription(
            String, "/delivery/balanced_messages", self.balance_message_callback,
            self.delivery_message_qos,
        )
        self._install_balance_filter()

    def _load_initial_sct(self):
        self.current_mission = "balanced_delivery"
        self._load_named_sct("balanced_delivery/supervisor.yaml")

    def _load_sct_from_yaml(self, config_path):
        super()._load_sct_from_yaml(config_path)
        required = {
            "EV_object_not_visible", "EV_object_visible", "EV_object_reached",
            "EV_zone_a_not_visible", "EV_zone_a_visible", "EV_zone_a_reached",
            "EV_zone_b_not_visible", "EV_zone_b_visible", "EV_zone_b_reached",
            "EV_pick_up_object", "EV_drop_zone_a", "EV_drop_zone_b",
        }
        available = set(self.sct.EV)
        missing = sorted(
            event for event in required
            if event not in available
            and STACK_EVENT_ALIASES.get(event) not in available
        )
        if missing:
            raise ValueError("Balanced YAML missing events: " + ", ".join(missing))

    def _install_uncontrollable_callbacks(self):
        for alias, source in STACK_EVENT_ALIASES.items():
            if alias not in self.sct.EV and source in self.sct.EV:
                self.sct.EV[alias] = self.sct.EV[source]
        super()._install_uncontrollable_callbacks()
        for event in self.sct.EV:
            if re.fullmatch(
                r"EV_(?:object|zone_[ab])_(?:not_visible|visible|reached|not_reached)", event
            ):
                self.sct.add_callback(
                    self.sct.EV[event], None,
                    lambda _data, event=event: self._consume_color_event(event), None,
                )
            elif event in {
                "EV_received_delivered_zone_a", "EV_received_delivered_zone_b"
            }:
                self.sct.add_callback(
                    self.sct.EV[event], None,
                    lambda _data, event=event: self._consume_delivery_event(event), None,
                )

    def _install_delivery_controllable_callbacks(self):
        uses_separate_drop_event = "EV_drop_object" in self.sct.EV

        def select_zone(zone):
            def callback(data):
                # Some balanced-delivery supervisors encode the completed drop
                # directly as EV_drop_zone_a/b.  Older supervisors select the
                # zone first and emit a separate EV_drop_object afterwards.
                self.selected_delivery_zone = None
                self._select_delivery_zone(zone)
                if not uses_separate_drop_event and self.selected_delivery_zone == zone:
                    # EV_drop_zone_a/b is enabled by the YAML only after the
                    # matching zone-reached observation.  Do not repeat that
                    # decision using a separately sampled world pose here.
                    self._confirm_selected_zone_delivery(data, require_position=False)

            return callback

        callbacks = {
            "EV_pick_up_object": self._confirm_generic_pickup,
            "EV_drop_zone_a": select_zone("zone_a"),
            "EV_drop_zone_b": select_zone("zone_b"),
            "EV_drop_object": self._confirm_selected_zone_delivery,
        }
        for event, callback in callbacks.items():
            if event in self.sct.EV:
                self.sct.add_callback(self.sct.EV[event], callback, None, None)

    def _task_action_spec(self, ev_name):
        ev_name = {
            "EV_search_color": "EV_search_zone",
            "EV_approach_color": "EV_approach_zone",
        }.get(ev_name, ev_name)
        return super()._task_action_spec(ev_name)

    def _install_balance_filter(self):
        unfiltered = self.sct.get_active_controllable_events

        def filtered():
            events = unfiltered()
            disabled = set()
            if self.balance_state == "zone_a_has_extra":
                disabled.add("EV_drop_zone_a")
            elif self.balance_state == "zone_b_has_extra":
                disabled.add("EV_drop_zone_b")
            if self.box_consumed:
                disabled.add("EV_pick_up_object")
            if not self._selected_zone_reached():
                disabled.add("EV_drop_object")
            for name in disabled:
                if name in self.sct.EV:
                    events[self.sct.EV[name]] = 0
            return events

        self.sct.get_active_controllable_events = filtered

    def color_event_callback(self, msg):
        match = SOURCE_EVENT_RE.fullmatch(msg.data.strip())
        if match is not None:
            color, kind, observation = (
                match.group("color"), match.group("kind"),
                match.group("observation")
            )
        else:
            match = COLOR_EVENT_RE.fullmatch(msg.data.strip())
            if match is None:
                return
            color = match.group("color")
            observation = match.group("observation")
            kind = "object" if color == "green" else "zone"
        if kind == "object":
            if self.generic_phase != "object" or self.box_consumed:
                return
            event = f"EV_object_{observation}"
            if observation != "not_visible":
                self.active_object_color = color
            key = "object"
        else:
            zone = {"red": "zone_a", "blue": "zone_b"}.get(color)
            if zone is None or self.generic_phase != "zone":
                return
            event = f"EV_{zone}_{observation}"
            if observation != "not_visible":
                self.active_zone_color = color
            key = zone
        if event not in self.sct.EV:
            return
        if self._motion_event_in_progress():
            self.deferred_color_events[key] = event
        else:
            self._accept_generic_event(event)

    def _accept_generic_event(self, event):
        prefix = event.rsplit("_", 1)[0]
        self.pending_color_events = {
            item for item in self.pending_color_events
            if not item.startswith(prefix + "_")
        }
        self.pending_color_events.add(event)

    def _release_deferred_color_events(self):
        if self._motion_event_in_progress():
            return
        events = list(self.deferred_color_events.values())
        self.deferred_color_events.clear()
        for event in events:
            self._accept_generic_event(event)

    def _delivery_target_approach_components(self, target):
        color = self.active_object_color if target == "object" else self.active_zone_color
        # The plain color detector publishes <color>_horizontal_offset.
        offset = self.color_horizontal_offsets.get(color, float("nan"))
        angular = -self.approach_steering_gain * offset if math.isfinite(offset) else 0.0
        return self.approach_linear_x, angular

    def _nearest_pickup_box(self):
        if self.world_robot_pose is None:
            return None
        position = self.world_robot_pose.translation
        nearest = None
        best_distance = float("inf")
        for name, (x, y, half_x, half_y) in self.pickup_boxes.items():
            if name in self.consumed_boxes:
                continue
            distance = math.hypot(max(abs(position.x - x) - half_x, 0.0),
                                  max(abs(position.y - y) - half_y, 0.0))
            if math.isfinite(distance) and distance <= best_distance:
                nearest, best_distance = name, distance
        return nearest

    def _confirm_generic_pickup(self, _data):
        if self.box_consumed:
            return
        box_name = self._nearest_pickup_box()
        if box_name is None:
            self.get_logger().warning("Pickup rejected: no available box")
            return
        if not (self.delete_entity_client.service_is_ready() and self.spawn_entity_client.service_is_ready()):
            return
        self.consumed_boxes.add(box_name)
        self.balance_message_pub.publish(String(data=f"{self.ns}:picked:{box_name}"))
        self.box_consumed = True
        self.selected_delivery_zone = None
        self.generic_phase = self.delivery_phase = "zone"
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        self._pick_up_assigned_box(box_name)

    def _pick_up_assigned_box(self, box_name):
        delete = DeleteEntity.Request()
        delete.entity.name = box_name
        delete.entity.type = Entity.MODEL
        self.delete_entity_client.call_async(delete)
        entity_name = f"{self.ns}_carried_generic_box"
        x, y, z, orientation = self._cargo_world_pose()
        spawn = SpawnEntity.Request()
        spawn.entity_factory.name = entity_name
        spawn.entity_factory.allow_renaming = False
        spawn.entity_factory.sdf = self._cargo_sdf(entity_name, "green")
        spawn.entity_factory.pose.position.x = x
        spawn.entity_factory.pose.position.y = y
        spawn.entity_factory.pose.position.z = z
        spawn.entity_factory.pose.orientation = orientation
        future = self.spawn_entity_client.call_async(spawn)
        future.add_done_callback(
            lambda completed: self._cargo_spawn_done(completed, entity_name)
        )

    def _zone_allowed(self, zone):
        return (
            self.balance_state == "equal"
            or self.balance_state == "zone_a_has_extra" and zone == "zone_b"
            or self.balance_state == "zone_b_has_extra" and zone == "zone_a"
        )

    def _apply_delivery(self, zone):
        self.balance_state = (
            f"{zone}_has_extra" if self.balance_state == "equal" else "equal"
        )

    def _record_team_delivery(self):
        if self.team_delivery_count >= TOTAL_DELIVERY_BOXES:
            return
        self.team_delivery_count += 1
        self.task_progress_pub.publish(Int32(data=self.team_delivery_count))
        if (
            self.team_delivery_count == TOTAL_DELIVERY_BOXES
            and not self.task_complete_sent
        ):
            self.all_colors_reached = True
            self.task_complete_sent = True
            self.task_complete_pub.publish(Bool(data=True))
            self.get_logger().info(
                "All six balanced-delivery boxes completed by the team."
            )

    def _select_delivery_zone(self, zone):
        if not self._zone_allowed(zone):
            self.get_logger().warning(f"Rejected unbalanced zone selection for {zone}")
            return
        self.selected_delivery_zone = zone
        self.active_zone_color = "red" if zone == "zone_a" else "blue"
        self.get_logger().info(f"Selected delivery destination {zone}")

    def _selected_zone_reached(self):
        if self.selected_delivery_zone is None or self.world_robot_pose is None:
            return False
        target = self.delivery_zones.get(self.selected_delivery_zone)
        if target is None:
            return False
        position = self.world_robot_pose.translation
        return math.hypot(position.x - target[0], position.y - target[1]) <= 1.0

    def _confirm_selected_zone_delivery(self, _data, require_position=True):
        zone = self.selected_delivery_zone
        if zone is None:
            self.get_logger().warning("Rejected drop: no delivery zone selected")
            return
        if not self._zone_allowed(zone):
            self.get_logger().warning(f"Rejected unbalanced drop for {zone}")
            return
        if require_position and not self._selected_zone_reached():
            self.get_logger().warning(f"Rejected drop outside {zone}")
            return
        if self.carried_box_entity is not None:
            self._delete_carried_box()
        self._apply_delivery(zone)
        self.local_delivery_count += 1
        self.box_consumed = (
            self.robot_index + self.local_delivery_count * self.total_robots
            >= TOTAL_DELIVERY_BOXES
        )
        self._record_team_delivery()
        self.task_progress_event_pub.publish(String(data=zone))
        self.balance_message_pub.publish(String(data=f"{self.ns}:delivered:{zone}"))
        self.selected_delivery_zone = None
        self.generic_phase = self.delivery_phase = "object"
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        self.get_logger().info(f"DELIVERED {zone}; balance={self.balance_state}")

    def balance_message_callback(self, msg):
        picked = re.fullmatch(r"robot_\d+:picked:(generic_delivery_box_\d+)", msg.data.strip())
        if picked:
            self.consumed_boxes.add(picked.group(1))
            return
        match = MESSAGE_RE.fullmatch(msg.data.strip())
        if match is None or match.group("sender") == self.ns:
            return
        zone = match.group("zone")
        if not self._zone_allowed(zone):
            self.get_logger().error(f"Out-of-balance remote delivery: {msg.data}")
            return
        self._apply_delivery(zone)
        self._record_team_delivery()
        event = f"EV_received_delivered_{zone}"
        if event in self.sct.EV:
            self.pending_delivery_events.add(event)
        self.get_logger().info(f"RECEIVED {zone}; balance={self.balance_state}")

    def timer_callback(self):
        self.robot_retired = False
        self.all_colors_reached = False
        super().timer_callback()


def main(args=None):
    rclpy.init(args=args)
    node = BalancedDeliverySupervisor()
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
