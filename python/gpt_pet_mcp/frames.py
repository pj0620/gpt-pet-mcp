"""Coordinate conversions between ROS 2 and the tool contract.

ROS (the `map` frame): metres, x and y on the floor, yaw in radians counter-clockwise from +x.
The contract (AI2-THOR style, what the brain and the copied modules speak): metres, floor
positions as `x`/`z` with `y` up, yaw in degrees clockwise from +z (90 = +x).

The mapping is `sim.x = ros.x`, `sim.z = ros.y`, `sim.yaw = (90 - degrees(theta)) mod 360`. It
is not a mirror: a contract yaw of 0 faces +z = +ros.y, 90 faces +x = +ros.x, and the yaw grows
clockwise seen from above in both pictures, so the copied modules (describe's bearings, the
perception geometry, the map overlays with their forward vector (sin yaw, cos yaw)) work
untouched. The rendered map has +ros.x to the right and +ros.y downward, as the simulator's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from gpt_pet_mcp import config


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def yaw_to_sim(theta_rad: float) -> float:
    """ROS yaw (radians, counter-clockwise from +x) -> contract yaw (degrees, clockwise from +z)."""
    return (90.0 - math.degrees(theta_rad)) % 360.0


def yaw_to_ros(yaw_deg: float) -> float:
    """Contract yaw -> ROS yaw in radians, wrapped to (-pi, pi]."""
    return math.radians(wrap_deg(90.0 - yaw_deg))


def to_sim_xz(x_ros: float, y_ros: float) -> Tuple[float, float]:
    return float(x_ros), float(y_ros)


def to_ros_xy(x_sim: float, z_sim: float) -> Tuple[float, float]:
    return float(x_sim), float(z_sim)


def yaw_of_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(theta_rad: float) -> Tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation about +z."""
    return 0.0, 0.0, math.sin(theta_rad / 2.0), math.cos(theta_rad / 2.0)


@dataclass(frozen=True)
class Pose2D:
    """The robot's pose in contract terms: floor position (x, z) and yaw in degrees."""

    x: float
    z: float
    yaw_deg: float

    @classmethod
    def from_ros(cls, x: float, y: float, theta_rad: float) -> "Pose2D":
        sx, sz = to_sim_xz(x, y)
        return cls(sx, sz, yaw_to_sim(theta_rad))

    @property
    def position(self) -> Dict[str, float]:
        return {"x": self.x, "y": 0.0, "z": self.z}

    @property
    def xz(self) -> Tuple[float, float]:
        return self.x, self.z

    @property
    def theta_ros(self) -> float:
        return yaw_to_ros(self.yaw_deg)

    def forward(self) -> Tuple[float, float]:
        """Unit forward vector in contract coordinates (sin yaw, cos yaw)."""
        rad = math.radians(self.yaw_deg)
        return math.sin(rad), math.cos(rad)

    def camera_position(
        self, height_m: float = config.CAMERA_HEIGHT_M, forward_m: float = config.CAMERA_FORWARD_M
    ) -> Dict[str, float]:
        fx, fz = self.forward()
        return {"x": self.x + forward_m * fx, "y": height_m, "z": self.z + forward_m * fz}

    def payload(self) -> Dict[str, float]:
        """The `agent` block of the contract: x/z to 2 dp, yaw to 1 dp."""
        return {"x": round(self.x, 2) + 0.0, "z": round(self.z, 2) + 0.0, "yaw": round(self.yaw_deg, 1) + 0.0}


def bearing_to(pose: Pose2D, target_xz: Tuple[float, float]) -> float:
    """Where a target lies relative to the robot's heading: degrees, positive to the right."""
    dx = target_xz[0] - pose.x
    dz = target_xz[1] - pose.z
    return wrap_deg(math.degrees(math.atan2(dx, dz)) - pose.yaw_deg)


def heading_toward(from_xz: Tuple[float, float], to_xz: Tuple[float, float]) -> float:
    """The contract yaw that faces from one floor point toward another."""
    return math.degrees(math.atan2(to_xz[0] - from_xz[0], to_xz[1] - from_xz[1])) % 360.0


def dist_xz(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
