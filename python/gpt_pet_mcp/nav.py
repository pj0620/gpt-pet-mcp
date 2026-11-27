"""nav2 goals behind the simulator's NavManager API.

`RobotNavManager.set_goal` plans with `compute_path_to_pose` (so a destination with no path fails
at once, like the simulator's planner), starts `navigate_to_pose` and returns immediately; a
progress thread integrates the distance driven from TF while Nav2 drives; `status()` has the
simulator's shape (state idle | planning | following | succeeded | canceled | failed, goal,
progress, agent, collisions, reason); `cancel` preempts. Drives to objects end OBJECT_STANDOFF_M
short of the object, facing it; drives to a person PERSON_STANDOFF_M short; passages are driven
to exactly.

Lock rule (the simulator's): _state_lock guards this manager's fields only and is never held
across a ROS call.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from loguru import logger
from nav2_msgs.action import ComputePathToPose, NavigateToPose


def kind_is_arrival(kind: str) -> bool:
    """Drives to objects and people end a standoff short of them, so the camera stopping the robot
    near the end is the arrival; a passage is driven through, so a stop there is a block."""
    return kind in ("object", "person")

from gpt_pet_mcp import config
from gpt_pet_mcp.attention import standoff_point
from gpt_pet_mcp.frames import Pose2D, dist_xz, heading_toward, quaternion_from_yaw, to_ros_xy, to_sim_xz, yaw_to_ros
from gpt_pet_mcp.ros_bridge import ActionCall, RosBridge


class NavError(ValueError):
    """Planning or starting a drive failed; the tool layer prefixes it with the destination."""


def _cum_from(corners: List[Dict[str, float]]) -> List[float]:
    """cum[i] = path length from corner i through the last corner."""
    cum = [0.0] * len(corners)
    for i in range(len(corners) - 2, -1, -1):
        cum[i] = cum[i + 1] + dist_xz((corners[i]["x"], corners[i]["z"]), (corners[i + 1]["x"], corners[i + 1]["z"]))
    return cum


def simplify_path(points: np.ndarray, tolerance_m: float = config.PATH_SIMPLIFY_M) -> np.ndarray:
    """Ramer-Douglas-Peucker: the corners of a dense Nav2 path."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= 2:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        a, b = pts[i0], pts[i1]
        ab = b - a
        length = np.hypot(*ab)
        seg = pts[i0 + 1 : i1]
        if length < 1e-9:
            dists = np.hypot(*(seg - a).T)
        else:
            dists = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / length
        j = int(np.argmax(dists))
        if dists[j] > tolerance_m:
            keep[i0 + 1 + j] = True
            stack.append((i0, i0 + 1 + j))
            stack.append((i0 + 1 + j, i1))
    return pts[keep]


@dataclass
class NavGoal:
    goal_id: str
    dest_id: str
    label: str
    kind: str  # object | passage | person
    target: Dict[str, float]  # where the destination is (contract frame)
    goal_xz: Tuple[float, float]  # where the robot drives to
    goal_yaw_deg: float  # which way it faces there (contract yaw)
    speed: float
    corners: List[Dict[str, float]] = field(default_factory=list)
    cum_from: List[float] = field(default_factory=list)
    cum_to: List[float] = field(default_factory=list)  # path length from the start to each corner
    length_m: float = 0.0


class RobotNavManager:
    def __init__(self, bridge: RosBridge) -> None:
        self._bridge = bridge
        self._state_lock = threading.Lock()
        self._phase = "idle"
        self._goal: Optional[NavGoal] = None
        self._reason: Optional[str] = None
        self._traveled = 0.0
        self._remaining: Optional[float] = None
        self._recoveries = 0
        self._wp_index = 0
        self._call: Optional[ActionCall] = None
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._blocker_source = None  # Phase 3/4: a callable naming what stands ahead

    def set_blocker_source(self, source) -> None:
        self._blocker_source = source

    def set_lane_source(self, source) -> None:
        """A callable -> metres to the nearest thing in the lane ahead by the Kinect (None = clear),
        polled while a drive runs: what the lidar plane misses (seats, bottles, low boxes)."""
        self._lane_source = source

    def _lane_ahead(self) -> Optional[float]:
        source = getattr(self, "_lane_source", None)
        if source is None or config.NAV_LANE_STOP_M <= 0.0:
            return None
        try:
            return source()
        except Exception:  # noqa: BLE001 - a camera hiccup must not end a drive
            return None

    # --- public API (the simulator's) ---

    def active(self) -> bool:
        with self._state_lock:
            return self._phase in ("planning", "following")

    def set_goal(self, dest_id: str, target: Dict[str, float], label: str, speed: float, kind: str = "object") -> dict:
        bridge = self._bridge
        preempted: Optional[str] = None
        with self._state_lock:
            if self._phase in ("planning", "following") and self._goal is not None:
                preempted = self._goal.goal_id
        self.cancel(wait=True)

        with self._state_lock:
            self._phase = "planning"
            self._reason = None
        try:
            goal = self._plan(dest_id, target, label, speed, kind)
        except NavError as exc:
            with self._state_lock:
                self._phase = "failed"
                self._reason = f"planning failed: {exc}"
            raise

        effective = min(speed, config.ROBOT_MAX_SPEED_MPS)
        bridge.publish_speed_limit(effective)
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = self._pose_stamped(goal.goal_xz, goal.goal_yaw_deg)
        call = bridge.call(bridge.navigate, nav_goal, config.NAVIGATE_ACTION)
        if not call.wait_accepted(config.ACTION_ACCEPT_TIMEOUT_S) or not call.accepted:
            reason = "Nav2 rejected the goal (is the navigation stack active?)"
            with self._state_lock:
                self._phase = "failed"
                self._reason = reason
            raise NavError(reason)

        cancel = threading.Event()
        with self._state_lock:
            self._goal = goal
            self._phase = "following"
            self._traveled = 0.0
            self._remaining = goal.length_m
            self._recoveries = 0
            self._wp_index = 0
            self._call = call
            self._cancel = cancel
        self._thread = threading.Thread(target=self._follow, args=(goal, call, cancel), name=f"nav-{goal.goal_id}", daemon=True)
        self._thread.start()
        logger.info(
            "Nav goal {} -> {} ({:.2f} m, {} waypoints, {:.2f} m/s)", goal.goal_id, dest_id, goal.length_m, len(goal.corners), effective
        )
        return {
            "goal_id": goal.goal_id,
            "destination_id": dest_id,
            "label": label,
            "state": "following",
            "path": {
                "waypoints": [[round(c["x"], 2), round(c["z"], 2)] for c in goal.corners],
                "length_m": round(goal.length_m, 2),
            },
            "speed_mps": effective,
            "eta_s": round(goal.length_m / effective, 1) if effective > 0 else 0.0,
            "preempted_goal_id": preempted,
        }

    def cancel(self, wait: bool = True, timeout: float = config.CANCEL_TIMEOUT_S) -> bool:
        with self._state_lock:
            call = self._call
            thread = self._thread
            running = self._phase in ("planning", "following")
        if not running or call is None:
            return False
        self._cancel.set()
        call.cancel(wait_s=timeout if wait else 0.0)
        self._bridge.publish_twist()
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("Nav thread did not stop within {} s", timeout)
        return True

    def preempt_for_manual(self) -> bool:
        return self.cancel(wait=True)

    def path_snapshot(self) -> Optional[List[Dict[str, float]]]:
        """Remaining corners of the active path, for map overlays."""
        with self._state_lock:
            if self._phase != "following" or self._goal is None:
                return None
            return [dict(c) for c in self._goal.corners[self._wp_index :]]

    def goal_target(self) -> Optional[Dict[str, float]]:
        with self._state_lock:
            if self._goal is None or self._phase not in ("planning", "following"):
                return None
            return dict(self._goal.target)

    def status(self) -> dict:
        with self._state_lock:
            phase = self._phase
            goal = self._goal
            traveled = self._traveled
            remaining = self._remaining
            recoveries = self._recoveries
            reason = self._reason
            wp_index = self._wp_index

        agent = None
        pose = self._bridge.pose()
        if pose is not None:
            agent = pose.payload()

        out = {
            "state": phase,
            "goal": None,
            "progress": None,
            "agent": agent,
            "collisions": recoveries,
            "reason": reason,
        }
        if goal is None:
            return out
        out["goal"] = {"goal_id": goal.goal_id, "destination_id": goal.dest_id, "label": goal.label}
        corners = goal.corners
        if phase == "succeeded" or not corners:
            remaining_m = 0.0
        elif remaining is not None:
            remaining_m = max(0.0, float(remaining))
        elif pose is not None:
            i = min(wp_index, len(corners) - 1)
            remaining_m = dist_xz(pose.xz, (corners[i]["x"], corners[i]["z"])) + goal.cum_from[i]
        else:
            remaining_m = max(0.0, goal.length_m - traveled)
        total = traveled + remaining_m
        out["progress"] = {
            "distance_traveled_m": round(traveled, 2),
            "distance_remaining_m": round(remaining_m, 2),
            "percent": 100 if phase == "succeeded" else int(100 * traveled / total) if total > 1e-6 else 0,
            "eta_s": round(remaining_m / goal.speed, 1) if goal.speed > 0 else 0.0,
            "waypoint": min(wp_index + 1, len(corners)) if corners else 0,
            "waypoints_total": len(corners),
        }
        return out

    # --- planning ---

    def _pose_stamped(self, xz: Tuple[float, float], yaw_deg: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = config.MAP_FRAME
        # stamp 0 = the latest transform, as bin/send_goal.py and bin/nav/nav_goal.py do
        x, y = to_ros_xy(*xz)
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        theta = yaw_to_ros(yaw_deg)
        qx, qy, qz, qw = quaternion_from_yaw(theta)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qx, qy, qz, qw
        return msg

    def _standoff(self, target_xz: Tuple[float, float], robot: Pose2D, kind: str) -> Tuple[float, float]:
        """Where a drive ends: passages exactly, objects and people a standoff short of them, on
        a free costmap cell when the costmap is known."""
        if kind == "passage":
            return target_xz
        standoff = config.PERSON_STANDOFF_M if kind == "person" else config.OBJECT_STANDOFF_M
        free = self._bridge.costmap.free_cells_xz()
        for distance in (standoff,) + tuple(config.STANDOFF_FALLBACKS_M):
            point = standoff_point(target_xz, robot.xz, distance, free)
            if point is not None:
                return (point["x"], point["z"])
        raise NavError(f"no free floor near ({target_xz[0]:.2f}, {target_xz[1]:.2f}) to stand on")

    def _plan(self, dest_id: str, target: Dict[str, float], label: str, speed: float, kind: str) -> NavGoal:
        bridge = self._bridge
        robot = bridge.pose()
        if robot is None:
            raise NavError("the robot's pose is unknown (no map->base_link transform: is slam_toolbox running?)")
        if not bridge.ready(bridge.compute_path):
            raise NavError(f"Nav2 is not running (/{config.COMPUTE_PATH_ACTION} unavailable)")
        target_xz = (float(target["x"]), float(target["z"]))
        goal_xz = self._standoff(target_xz, robot, kind)
        # objects and people: arrive facing them; passages: face the way in (from where the robot is)
        yaw = heading_toward(goal_xz, target_xz) if kind != "passage" else heading_toward(robot.xz, goal_xz)
        if dist_xz(goal_xz, target_xz if kind != "passage" else robot.xz) < 1e-6:
            yaw = robot.yaw_deg

        request = ComputePathToPose.Goal()
        request.goal = self._pose_stamped(goal_xz, yaw)
        request.use_start = False
        call = bridge.call(bridge.compute_path, request, config.COMPUTE_PATH_ACTION)
        if not call.wait(config.PLAN_TIMEOUT_S):
            call.cancel(1.0)
            raise NavError(f"the planner did not answer within {config.PLAN_TIMEOUT_S:g} s")
        if call.status != GoalStatus.STATUS_SUCCEEDED or call.result is None or not call.result.path.poses:
            raise NavError(f"no path to ({goal_xz[0]:.2f}, {goal_xz[1]:.2f})")
        pts_ros = np.array([[p.pose.position.x, p.pose.position.y] for p in call.result.path.poses], dtype=np.float64)
        pts = np.array([to_sim_xz(x, y) for x, y in pts_ros])
        length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
        corners_xy = simplify_path(pts)
        if len(corners_xy) > 1:
            corners_xy = corners_xy[1:]  # the first point is where the robot stands
        corners = [{"x": float(x), "y": 0.0, "z": float(z)} for x, z in corners_xy]
        cum_from = _cum_from(corners)
        cum_to = [0.0] * len(corners)
        prev = robot.xz
        acc = 0.0
        for i, c in enumerate(corners):
            acc += dist_xz(prev, (c["x"], c["z"]))
            cum_to[i] = acc
            prev = (c["x"], c["z"])
        return NavGoal(
            goal_id=uuid.uuid4().hex[:8],
            dest_id=dest_id,
            label=label,
            kind=kind,
            target=dict(target),
            goal_xz=goal_xz,
            goal_yaw_deg=yaw,
            speed=min(speed, config.ROBOT_MAX_SPEED_MPS),
            corners=corners,
            cum_from=cum_from,
            cum_to=cum_to,
            length_m=length,
        )

    # --- following ---

    def _finish(self, phase: str, reason: Optional[str] = None) -> None:
        with self._state_lock:
            self._phase = phase
            self._reason = reason
            self._remaining = 0.0 if phase == "succeeded" else self._remaining
        if reason:
            logger.info("Nav finished: {} ({})", phase, reason)
        else:
            logger.info("Nav finished: {}", phase)

    def _follow(self, goal: NavGoal, call: ActionCall, cancel: threading.Event) -> None:
        try:
            self._run_follow(goal, call, cancel)
        except Exception as exc:  # noqa: BLE001 - never die silently
            logger.exception("Nav goal {} crashed", goal.goal_id)
            self._finish("failed", f"internal error: {exc}"[:120])

    def _run_follow(self, goal: NavGoal, call: ActionCall, cancel: threading.Event) -> None:
        bridge = self._bridge
        last = bridge.pose()
        lane_hit: Optional[float] = None
        lane_check_at = 0.0
        while not call.done:
            if cancel.wait(config.PROGRESS_SAMPLE_S):
                if not call.wait(config.CANCEL_TIMEOUT_S):
                    break
                continue
            now = time.monotonic()
            if now - lane_check_at >= config.LANE_CHECK_PERIOD_S:
                lane_check_at = now
                ahead = self._lane_ahead()
                if ahead is not None and ahead < config.NAV_LANE_STOP_M:
                    lane_hit = ahead
                    logger.info("Nav goal {}: the Kinect sees something {:.2f} m ahead in the lane; stopping", goal.goal_id, ahead)
                    call.cancel(config.CANCEL_TIMEOUT_S)
                    bridge.stop_robot(5)
                    break
            pose = bridge.pose()
            feedback = call.feedback
            with self._state_lock:
                if pose is not None and last is not None:
                    step = dist_xz(pose.xz, last.xz)
                    if step <= config.PROGRESS_JUMP_CAP_M:
                        self._traveled += step
                    # the corner the robot is heading for: the first one its progress has not passed
                    while self._wp_index < len(goal.cum_to) - 1 and self._traveled > goal.cum_to[self._wp_index] + 0.05:
                        self._wp_index += 1
                if feedback is not None:
                    self._remaining = float(feedback.distance_remaining)
                    self._recoveries = int(feedback.number_of_recoveries)
            if pose is not None:
                last = pose

        status = call.status
        pose = bridge.pose()
        where = f" near ({pose.x:.2f}, {pose.z:.2f})" if pose is not None else ""
        if lane_hit is not None:
            # Close to the destination this is the arrival (the camera stops the robot a safe
            # standoff short of a seat or a low object the lidar never saw); earlier in the path it
            # is something in the way.
            with self._state_lock:
                remaining = self._remaining
            to_target = dist_xz(pose.xz, (goal.target["x"], goal.target["z"])) if pose is not None else None
            near_target = to_target is not None and to_target <= config.LANE_ARRIVAL_TARGET_M
            if kind_is_arrival(goal.kind) and near_target and remaining is not None and remaining <= config.LANE_ARRIVAL_REMAINING_M:
                self._finish("succeeded")
            else:
                blocker = self._blocker()
                by = f" by {blocker}" if blocker else ""
                self._finish("failed", f"blocked{by}{where} (seen by the camera {lane_hit:.2f} m ahead)")
            return
        if call.error:
            self._finish("failed", f"internal error: {call.error}"[:120])
        elif status == GoalStatus.STATUS_SUCCEEDED:
            self._finish("succeeded")
        elif status == GoalStatus.STATUS_CANCELED or cancel.is_set():
            self._finish("canceled")
        elif status == ActionCall.REJECTED:
            self._finish("failed", "Nav2 rejected the goal (is the navigation stack active?)")
        else:
            with self._state_lock:
                recoveries = self._recoveries
            blocker = self._blocker()
            by = f" by {blocker}" if blocker else ""
            after = f" after {recoveries} recoveries" if recoveries else ""
            self._finish("failed", f"blocked{by}{where}{after}")

    def _blocker(self) -> Optional[str]:
        source = self._blocker_source
        if source is None:
            return None
        try:
            return source()
        except Exception:  # noqa: BLE001 - a broken source must not hide the outcome
            return None
