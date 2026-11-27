"""Manual control: do_move and do_rotate on the robot, with the simulator's result texts.

Ahead and back go through Nav2's `drive_on_heading` and `backup` behaviours: closed on odometry,
collision-checked against the local costmap, and reporting how far they got when something was
in the way. Spins go through `spin`. The mecanum base can strafe, which Nav2's behaviours
cannot, so a sideways move streams `/cmd_vel` directly at STRAFE_SPEED_MPS, measures the lateral
displacement on TF odom->base_link and stops at the distance, the time cap, or a lidar return
inside STRAFE_CLEAR_M on the moving side.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from loguru import logger
from nav2_msgs.action import BackUp, DriveOnHeading, Spin

from gpt_pet_mcp import config
from gpt_pet_mcp.ros_bridge import ActionCall, RosBridge


def _wrap(angle: float) -> float:
    """Radians wrapped to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

_DIRECTION = {"MoveAhead": "ahead", "MoveBack": "back", "MoveLeft": "to the left", "MoveRight": "to the right"}


def _duration(seconds: float) -> DurationMsg:
    msg = DurationMsg()
    whole = int(seconds)
    msg.sec = whole
    msg.nanosec = int((seconds - whole) * 1e9)
    return msg


class Moves:
    def __init__(self, bridge: RosBridge, blocker_source: Optional[Callable[[float], Optional[str]]] = None) -> None:
        self._bridge = bridge
        self._blocker_source = blocker_source  # bearing_deg -> name of what is there, or None

    def set_blocker_source(self, source: Optional[Callable[[float], Optional[str]]]) -> None:
        self._blocker_source = source

    def _blocker(self, bearing_deg: float) -> str:
        if self._blocker_source is not None:
            try:
                name = self._blocker_source(bearing_deg)
                if name:
                    return name
            except Exception:  # noqa: BLE001
                pass
        return "an obstacle"

    # --- do_move ---

    def move(self, action: str, magnitude: float) -> str:
        magnitude = max(0.0, float(magnitude))
        direction = _DIRECTION.get(action, action)
        if magnitude < 1e-3:
            return f"Successfully moved 0.00 m {direction}"
        if action in ("MoveLeft", "MoveRight"):
            sign = +1.0 if action == "MoveLeft" else -1.0
            if config.SIDESTEP_MODE == "strafe":
                return self._strafe(sign, magnitude, direction)
            return self._sidestep_by_turning(sign, magnitude, direction)
        if action == "MoveAhead":
            client, goal = self._bridge.drive, DriveOnHeading.Goal()
            name = config.DRIVE_ACTION
        else:
            client, goal = self._bridge.backup, BackUp.Goal()
            name = config.BACKUP_ACTION
        if not self._bridge.ready(client):
            return f"Move blocked: the robot's drive is not available (/{name} unavailable)"
        speed = config.ROBOT_MAX_SPEED_MPS
        goal.target.x = magnitude
        goal.speed = float(speed)
        goal.time_allowance = _duration(magnitude / speed * 1.5 + 3.0)
        call = self._bridge.call(client, goal, name)
        if not call.wait_accepted(config.ACTION_ACCEPT_TIMEOUT_S) or not call.accepted:
            return "Move blocked: Nav2 rejected the move (is the navigation stack active?)"
        call.wait(magnitude / speed * 1.5 + 10.0)
        if not call.done:
            call.cancel(2.0)
        moved = 0.0
        if call.feedback is not None:
            moved = float(call.feedback.distance_traveled)
        if call.status == GoalStatus.STATUS_SUCCEEDED:
            return f"Successfully moved {magnitude:.2f} m {direction}"
        if call.error:
            return f"Move blocked: {call.error}"
        blocker = self._blocker(0.0 if action == "MoveAhead" else 180.0)
        if moved > 0.01:
            return f"Moved {moved:.2f} of {magnitude:.2f} m {direction}, then blocked by {blocker}"
        return f"Move blocked by {blocker}: did not move"

    def _sidestep_by_turning(self, sign: float, magnitude: float, direction: str) -> str:
        """MoveLeft/MoveRight without strafing: turn 90 degrees to that side, drive, turn back.
        The drive's own texts carry over with the side-step's words ("to the left")."""
        out, back = ("RotateLeft", "RotateRight") if sign > 0 else ("RotateRight", "RotateLeft")
        start = self._bridge.odom_pose()
        turned = self.rotate(out, 90.0)
        if not turned.startswith("Successfully"):
            return f"Move blocked by an obstacle: could not turn to face {direction} ({turned})"
        drove = self.move("MoveAhead", magnitude)
        returned = self.rotate(back, 90.0)
        if start is not None:
            self._square_up(start[2])  # the two spins overshoot alike, but not exactly
        text = drove.replace(" ahead", f" {direction}")
        if not returned.startswith("Successfully"):
            text += f" (heading not restored: {returned})"
        logger.info("Side-step {} {:.2f} m by turning: {} / {} / {}", direction, magnitude, turned, drove, returned)
        return text

    def _strafe(self, sign: float, magnitude: float, direction: str) -> str:
        bridge = self._bridge
        deadline = time.monotonic() + 8.0
        while bridge.cmd_vel_subscribers() < 1 and time.monotonic() < deadline:
            time.sleep(0.2)
        if bridge.cmd_vel_subscribers() < 1:
            return "Move blocked: the robot's drive is not connected (no /cmd_vel subscriber)"
        start = bridge.odom_pose()
        if start is None:
            return "Move blocked: the robot's odometry is unknown (no odom->base_link transform)"
        x0, y0, theta0 = start
        left = (-math.sin(theta0), math.cos(theta0))  # the robot's +y at the start, in odom
        side_deg = 90.0 if sign > 0 else -90.0  # lidar sector on the moving side, robot frame
        blocker: Optional[str] = None
        moved = 0.0
        err = 0.0  # heading error (rad, +ccw) against the start, from the odom yaw
        max_err = 0.0
        kp, max_wz = config.STRAFE_YAW_KP, config.STRAFE_YAW_MAX_WZ
        abort = math.radians(config.STRAFE_YAW_ABORT_DEG)
        ratio = max(0.2, config.STRAFE_SLIP_RATIO)  # odometry metres per real metre sideways
        target = magnitude * ratio
        started = time.monotonic()

        def lateral(pose: tuple) -> float:
            """Real sideways travel estimate: odometry projected on the start's left axis, de-slipped."""
            return abs((pose[0] - x0) * left[0] + (pose[1] - y0) * left[1]) / ratio

        try:
            while True:
                if self._clearance(side_deg) < config.STRAFE_CLEAR_M:
                    blocker = self._blocker(-side_deg)  # bearing: positive to the right
                    break
                pose = bridge.odom_pose()
                wz = 0.0
                if pose is not None:
                    err = _wrap(pose[2] - theta0)
                    max_err = max(max_err, abs(err))
                    if abs(err) > abort:
                        blocker = f"heading drift ({math.degrees(err):+.0f} degrees)"
                        break
                    if kp > 0.0:
                        wz = max(-max_wz, min(max_wz, -kp * err))
                bridge.publish_twist(vy=sign * config.STRAFE_SPEED_MPS, wz=wz)
                time.sleep(0.05)
                pose = bridge.odom_pose()
                if pose is not None:
                    moved = lateral(pose)
                if moved >= magnitude:
                    break
                if time.monotonic() - started > config.STRAFE_MAX_S:
                    blocker = blocker or "an obstacle"
                    break
        finally:
            bridge.stop_robot(5)
        if kp > 0.0:
            self._square_up(theta0)
        pose = bridge.odom_pose()
        if pose is not None:
            moved = lateral(pose)
            err = _wrap(pose[2] - theta0)
        logger.info("Strafe {} {:.2f} m: moved {:.2f} m, heading error peak {:.1f} deg, final {:.1f} deg{}", direction, magnitude, moved,
                    math.degrees(max_err), math.degrees(err), f", blocked by {blocker}" if blocker else "")
        if blocker is None:
            return f"Successfully moved {moved:.2f} m {direction}"
        if moved > 0.01:
            return f"Moved {moved:.2f} of {magnitude:.2f} m {direction}, then blocked by {blocker}"
        return f"Move blocked by {blocker}: did not move"

    def _square_up(self, theta0: float) -> None:
        """Turn on the spot until the odom yaw is back within STRAFE_SQUARE_DEG of theta0."""
        bridge = self._bridge
        tol = math.radians(config.STRAFE_SQUARE_DEG)
        deadline = time.monotonic() + config.STRAFE_SQUARE_MAX_S
        try:
            while time.monotonic() < deadline:
                pose = bridge.odom_pose()
                if pose is None:
                    return
                err = _wrap(pose[2] - theta0)
                if abs(err) <= tol:
                    return
                wz = -config.STRAFE_YAW_KP * err
                wz = math.copysign(max(0.25, min(config.STRAFE_YAW_MAX_WZ, abs(wz))), wz)  # above the drive's dead band
                bridge.publish_twist(wz=wz)
                time.sleep(0.05)
        finally:
            bridge.stop_robot(5)

    def _clearance(self, side_deg: float) -> float:
        """Nearest lidar return (m) inside a STRAFE_SECTOR_DEG sector around `side_deg` of the robot
        frame (counter-clockwise positive, 0 = ahead); inf without a recent scan."""
        scan = self._bridge.scan(max_age_s=1.0)
        if scan is None:
            return math.inf
        offset = self._bridge.frame_offset(config.BASE_FRAME, scan.frame_id) if scan.frame_id else None
        yaw = offset[2] if offset is not None else 0.0
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment + yaw
        ranges = scan.ranges
        valid = np.isfinite(ranges) & (ranges > max(scan.range_min, 0.05)) & (ranges < scan.range_max)
        rel = (np.degrees(angles) - side_deg + 180.0) % 360.0 - 180.0
        sector = valid & (np.abs(rel) <= config.STRAFE_SECTOR_DEG / 2.0)
        if not sector.any():
            return math.inf
        return float(np.min(ranges[sector]))

    # --- do_rotate ---

    def rotate(self, action: str, degrees: float) -> str:
        degrees = float(degrees)
        if not self._bridge.ready(self._bridge.spin):
            return f"Rotate blocked: the robot's drive is not available (/{config.SPIN_ACTION} unavailable)"
        sign = -1.0 if action == "RotateRight" else 1.0  # ROS yaw grows counter-clockwise
        # Nav2's Spin stops late by a roughly constant angle on this robot; ask for less.
        commanded = degrees if degrees <= config.ROTATE_MIN_DEG else max(config.ROTATE_MIN_DEG, degrees - config.ROTATE_OVERSHOOT_DEG)
        goal = Spin.Goal()
        goal.target_yaw = float(sign * math.radians(commanded))
        goal.time_allowance = _duration(abs(math.radians(degrees)) / 0.4 * 1.5 + 3.0)
        call = self._bridge.call(self._bridge.spin, goal, config.SPIN_ACTION)
        if not call.wait_accepted(config.ACTION_ACCEPT_TIMEOUT_S) or not call.accepted:
            return "Rotate blocked: Nav2 rejected the spin (is the navigation stack active?)"
        call.wait(abs(math.radians(degrees)) / 0.4 * 1.5 + 10.0)
        if not call.done:
            call.cancel(2.0)
        if call.status == GoalStatus.STATUS_SUCCEEDED:
            return "Successfully rotated"
        turned = abs(math.degrees(float(call.feedback.angular_distance_traveled))) if call.feedback is not None else 0.0
        detail = call.error or f"{call.status_name}: something is in the way"
        return f"Rotate blocked: turned {turned:.0f} of {abs(degrees):.0f} degrees ({detail})"
