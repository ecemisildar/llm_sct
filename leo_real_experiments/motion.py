"""Shared motion types and geometry helpers for robot supervisors."""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionSpec:
    """How a controllable event is executed by a ROS supervisor."""

    linear_x: float = 0.0
    angular_z: float = 0.0
    hold_s: Optional[float] = None
    is_full_rotate: bool = False


def wrap_to_pi(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def yaw_from_quat(quaternion) -> float:
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
