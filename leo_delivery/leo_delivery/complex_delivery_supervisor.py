"""Nine-box delivery runtime using the existing delivery SCT event alphabet."""

from __future__ import annotations

import re
from typing import Optional

import rclpy
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import DeleteEntity, SetEntityPose, SpawnEntity
from std_msgs.msg import Bool, Int32, String

from leo_delivery.robot_supervisor import RobotSupervisor, TARGET_COLORS


BOXES_PER_COLOR = 3
PAYLOAD_CAPACITY = {"red": 1, "green": 2, "blue": 3}


class ComplexDeliverySupervisor(RobotSupervisor):
    """Deliver three boxes per color in payload-limited batches.

    Box identity and message quantities are runtime data.  SCT continues to see
    only the original color-level claim, pickup, drop, and observation events.
    """

    def __init__(self):
        super().__init__()
        self.delivered_box_counts = {color: 0 for color in TARGET_COLORS}
        self.carried_box_entities: list[str] = []
        self.carried_box_delete_requested: set[str] = set()
        self.carried_box_pose_request_pending: set[str] = set()
        self.current_load_count = 0
        self.current_load_color: Optional[str] = None

    def delivery_message_callback(self, msg: String):
        fields = msg.data.strip().split(":")
        if len(fields) == 4 and fields[1].lower() == "batch":
            sender, _, color, raw_quantity = fields
            color = color.lower()
            try:
                quantity = int(raw_quantity)
            except ValueError:
                return
            if (
                sender != self.ns
                and re.fullmatch(r"robot_(\d+)", sender) is not None
                and color in self.delivery_colors
                and quantity > 0
            ):
                self._record_delivered_boxes(
                    color, quantity, publish_event=False
                )
            return
        super().delivery_message_callback(msg)

    def _publish_delivery_message(
        self, action: str, color: str, quantity: int = 1
    ):
        if action != "batch":
            super()._publish_delivery_message(action, color)
            return
        self.delivery_message_pub.publish(
            String(data=f"{self.ns}:batch:{color}:{quantity}")
        )
        self.get_logger().info(
            f"DELIVERY PUBLISHED: batch:{color}:{quantity}"
        )

    def _record_delivered_boxes(
        self, color: str, quantity: int, publish_event: bool
    ):
        previous = self.delivered_box_counts[color]
        self.delivered_box_counts[color] = min(
            BOXES_PER_COLOR, previous + max(0, int(quantity))
        )
        accepted = self.delivered_box_counts[color] - previous
        if accepted == 0:
            return
        self.task_progress_pub.publish(
            Int32(data=sum(self.delivered_box_counts.values()))
        )
        if publish_event:
            for _ in range(accepted):
                self.task_progress_event_pub.publish(String(data=color))
        if (
            all(
                self.delivered_box_counts[color] == BOXES_PER_COLOR
                for color in self.delivery_colors
            )
            and not self.task_complete_sent
        ):
            self.all_colors_reached = True
            self.task_complete_sent = True
            self.task_complete_pub.publish(Bool(data=True))
            self.get_logger().info("All nine delivery boxes completed by the team.")

    def _delivery_message_received(self, color: str):
        # The preceding batch message already carries the exact box quantity.
        return

    def _delivery_message_published(self, color: str):
        # Quantity progress is recorded before the completion message.
        return

    def _confirm_pickup(self, _sup_data):
        color = self.active_delivery_color
        if color is not None:
            self.current_load_color = color
            remaining = BOXES_PER_COLOR - self.delivered_box_counts.get(color, 0)
            load_quantity = min(PAYLOAD_CAPACITY.get(color, 1), remaining)
            for _ in range(load_quantity):
                self.current_load_count += 1
                self._pick_up_box_model(color)
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        # The unchanged delivery plant permits one pickup followed by one
        # drop. One complex-task pickup therefore loads the full color batch.
        self.delivery_phase = "zone"

    def _confirm_delivery(self, _sup_data):
        quantity = self.current_load_count
        if self.carried_box_entities:
            self._delete_carried_boxes()
        if self.active_delivery_color is not None and quantity:
            color = self.active_delivery_color
            self._record_delivered_boxes(color, quantity, publish_event=True)
            # Batch progress and final completion are runtime messages, not
            # SCT events.
            self._publish_delivery_message("batch", color, quantity)
            if self.delivered_box_counts[color] == BOXES_PER_COLOR:
                # This robot has completed the color it claimed. Notify the
                # team using the existing delivered event, then retire this
                # robot immediately instead of waiting for all nine boxes.
                self._publish_delivery_message("delivered", color)
                self.active_delivery_color = None
                self._retire_robot_after_delivery()
        self.current_load_count = 0
        self.current_load_color = None
        self.pending_color_events.clear()
        self.deferred_color_events.clear()
        self.delivery_phase = "object"

    def _pick_up_box_model(self, color: str):
        if not (
            self.delete_entity_client.service_is_ready()
            and self.spawn_entity_client.service_is_ready()
        ):
            self.get_logger().error("Cargo create/remove services are not ready.")
            return
        box_index = (
            self.delivered_box_counts[color] + self.current_load_count
        )
        ground_box = DeleteEntity.Request()
        ground_box.entity.name = (
            f"{color}_delivery_box"
            if box_index == 1
            else f"{color}_delivery_box_{box_index}"
        )
        ground_box.entity.type = Entity.MODEL
        self.delete_entity_client.call_async(ground_box)

        entity_name = f"{self.ns}_carried_{color}_box_{box_index}"
        cargo_x, cargo_y, cargo_z, cargo_orientation = self._cargo_world_pose()
        request = SpawnEntity.Request()
        request.entity_factory.name = entity_name
        request.entity_factory.allow_renaming = False
        request.entity_factory.sdf = self._cargo_sdf(entity_name, color)
        request.entity_factory.pose.position.x = cargo_x
        request.entity_factory.pose.position.y = cargo_y
        request.entity_factory.pose.position.z = cargo_z + 0.27 * (
            self.current_load_count - 1
        )
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
        self.carried_box_entities.append(entity_name)
        self.get_logger().info(f"Cargo '{entity_name}' placed on robot.")
        self._update_carried_box_pose()

    def _delete_carried_boxes(self):
        if not self.delete_entity_client.service_is_ready():
            self.get_logger().warning("Cargo remove service is not ready; will retry.")
            return
        for entity_name in list(self.carried_box_entities):
            if entity_name in self.carried_box_delete_requested:
                continue
            request = DeleteEntity.Request()
            request.entity.name = entity_name
            request.entity.type = Entity.MODEL
            self.carried_box_delete_requested.add(entity_name)
            future = self.delete_entity_client.call_async(request)
            future.add_done_callback(
                lambda completed, entity_name=entity_name:
                    self._cargo_delete_done(completed, entity_name)
            )

    def _cargo_delete_done(self, future, entity_name: str):
        self.carried_box_delete_requested.discard(entity_name)
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - middleware failure
            self.get_logger().error(f"Cargo remove request failed: {exc}")
            return
        if not response.success:
            self.get_logger().warning(f"Simulator rejected removal of '{entity_name}'.")
            return
        if entity_name in self.carried_box_entities:
            self.carried_box_entities.remove(entity_name)

    def _update_carried_box_pose(self):
        if not self.set_pose_client.service_is_ready():
            return
        cargo_x, cargo_y, cargo_z, cargo_orientation = self._cargo_world_pose()
        for stack_index, entity_name in enumerate(self.carried_box_entities):
            if (
                entity_name in self.carried_box_delete_requested
                or entity_name in self.carried_box_pose_request_pending
            ):
                continue
            request = SetEntityPose.Request()
            request.entity.name = entity_name
            request.entity.type = Entity.MODEL
            request.pose.position.x = cargo_x
            request.pose.position.y = cargo_y
            request.pose.position.z = cargo_z + 0.27 * stack_index
            request.pose.orientation = cargo_orientation
            self.carried_box_pose_request_pending.add(entity_name)
            future = self.set_pose_client.call_async(request)
            future.add_done_callback(
                lambda completed, entity_name=entity_name:
                    self._carried_box_pose_done(completed, entity_name)
            )

    def _carried_box_pose_done(self, future, entity_name: str):
        self.carried_box_pose_request_pending.discard(entity_name)
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - middleware failure
            self.get_logger().error(f"Cargo pose request failed: {exc}")
            return
        if not response.success:
            self.get_logger().warning(
                f"Simulator rejected cargo pose for '{entity_name}'."
            )


def main(args=None):
    rclpy.init(args=args)
    node = ComplexDeliverySupervisor()
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
