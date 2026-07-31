"""Shared color-target approach behavior for task supervisors."""

import math

from std_msgs.msg import Float32


TARGET_COLORS = ("red", "green", "blue")


class TargetApproachMixin:
    """Own target offsets, subscriptions, and approach command calculation."""

    def _initialize_target_approach(self):
        self.color_horizontal_offsets = {
            color: float("nan") for color in TARGET_COLORS
        }
        self.approach_steering_gain = float(
            self.declare_parameter("approach_steering_gain", 1.2).value
        )
        self.approach_linear_x = float(
            self.declare_parameter("approach_linear_x", 0.20).value
        )

    def _create_target_approach_subscriptions(self):
        self.color_offset_subscriptions = [
            self.create_subscription(
                Float32,
                f"{color}_horizontal_offset",
                lambda message, color=color: self._color_offset_callback(
                    color, message
                ),
                10,
            )
            for color in TARGET_COLORS
        ]

    def _color_offset_callback(self, color: str, message: Float32):
        self.color_horizontal_offsets[color] = float(message.data)

    def _target_approach_components(self, event_name: str):
        color = event_name.removeprefix("EV_approach_")
        offset = self.color_horizontal_offsets.get(color, float("nan"))
        angular_z = (
            -self.approach_steering_gain * offset
            if math.isfinite(offset)
            else 0.0
        )
        return self.approach_linear_x, angular_z
