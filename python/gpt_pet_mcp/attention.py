"""Attention: the robot keeps a person framed on its own (gptpet-plans: People Interaction Design,
Attention Behaviour).

The pure helpers and the constants are ai2thor-mcp's `attention.py`, copied, so the robot uses the
same geometry and thresholds. `RobotAttentionManager` is the simulator's AttentionManager over the
robot: it sees the person through the pose detector on the Kinect frames (inside the frame, with
depth: the limits a detector has), aims with `/cmd_vel`, tilts the Kinect, approaches through the
navigation manager and backs up through Nav2's BackUp behaviour. Until GPTPET_MCP_ATTENTION=1 the
tools see `NullAttention`: attention is always off and the three attention tools are not served.

States: off | aiming | approaching | framed | searching | lost. Lock rule: _state_lock guards the
manager's fields only and is never held across a ROS call.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from gpt_pet_mcp import config
from gpt_pet_mcp.contract import attention_off

ACTIVE_STATES = ("aiming", "approaching", "framed", "searching")
STATES = ("off",) + ACTIVE_STATES + ("lost",)
SNAP_MAX_M = 0.4  # the standoff point must have a reachable cell this close
STANDOFF_RANGE_M = (0.7, 2.0)  # set_attention's standoff_m
TOO_CLOSE_FLOOR_M = 0.5  # never back away from a person farther than this (bodies touch at ~0.36 m)
FOLLOW_MAX_REPLANS = 1000  # following: re-plan as often as the person moves on
WIGGLE = (("RotateLeft", 15.0), ("RotateRight", 30.0), ("RotateLeft", 15.0))  # greet: hello
WIGGLE_PAUSE_S = 0.2


class AttentionError(ValueError):
    """A bad request: no person in the scene, or a destination that is not the person."""


# --- pure helpers (ai2thor-mcp) ---------------------------------------------------------------------


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def bearing_deg(robot_xz: Tuple[float, float], robot_yaw_deg: float, target_xz: Tuple[float, float]) -> float:
    """Where the target is relative to the robot's heading, degrees, clockwise positive (to the
    right), like describe.relative_bearing."""
    dx = target_xz[0] - robot_xz[0]
    dz = target_xz[1] - robot_xz[1]
    return wrap_deg(math.degrees(math.atan2(dx, dz)) - robot_yaw_deg)


def turn_for(bearing: float, tol_deg: Optional[float] = None, max_turn_deg: Optional[float] = None) -> Optional[Tuple[str, float]]:
    """The rotate step that centres a bearing, or None when it is within tolerance."""
    tol = config.ATTENTION_CENTER_TOL_DEG if tol_deg is None else tol_deg
    cap = config.ATTENTION_MAX_TURN_DEG if max_turn_deg is None else max_turn_deg
    if abs(bearing) <= tol:
        return None
    return ("RotateRight" if bearing > 0 else "RotateLeft", round(min(abs(bearing), cap), 1))


def tilt_wanted(distance_m: float, up_now: bool) -> bool:
    """Camera up below ATTENTION_NEAR_M, level beyond ATTENTION_LEVEL_M, unchanged in between."""
    if distance_m < config.ATTENTION_NEAR_M:
        return True
    if distance_m > config.ATTENTION_LEVEL_M:
        return False
    return up_now


def speed_mps(history: Sequence[Tuple[float, float, float]], now: float, window_s: float = 1.0) -> Optional[float]:
    """The person's speed over the last window, from (t, x, z) sightings; None without enough."""
    recent = [s for s in history if now - s[0] <= window_s]
    if len(recent) < 2:
        return None
    (t0, x0, z0), (t1, x1, z1) = recent[0], recent[-1]
    if t1 - t0 < 0.3:
        return None
    return math.hypot(x1 - x0, z1 - z0) / (t1 - t0)


def is_still(history: Sequence[Tuple[float, float, float]], now: float) -> bool:
    """True when the person has been in sight for ATTENTION_STILL_S and stayed within
    ATTENTION_STILL_MPS * ATTENTION_STILL_S of one spot over that time."""
    still_s = config.ATTENTION_STILL_S
    recent = [s for s in history if now - s[0] <= still_s]
    if len(recent) < 2 or now - recent[0][0] < still_s * 0.8:
        return False
    xs = [x for _, x, _ in recent]
    zs = [z for _, _, z in recent]
    return math.hypot(max(xs) - min(xs), max(zs) - min(zs)) < config.ATTENTION_STILL_MPS * still_s


def standoff_point(
    person_xz: Tuple[float, float],
    robot_xz: Tuple[float, float],
    standoff_m: float,
    reachable_xz: Any,
) -> Optional[Dict[str, float]]:
    """Where to stand: the person pulled standoff_m toward the robot, snapped to the nearest
    reachable cell (within SNAP_MAX_M); None when no cell is near, e.g. a person against a wall."""
    px, pz = person_xz
    dx, dz = robot_xz[0] - px, robot_xz[1] - pz
    dist = math.hypot(dx, dz)
    if dist < 1e-6:
        return None
    pull = min(standoff_m, dist)
    tx, tz = px + dx / dist * pull, pz + dz / dist * pull
    pts = np.asarray(reachable_xz if reachable_xz is not None else [], dtype=np.float64)
    if len(pts) == 0:
        return {"x": tx, "y": 0.0, "z": tz}
    gaps = np.hypot(pts[:, 0] - tx, pts[:, 1] - tz)
    i = int(np.argmin(gaps))
    if gaps[i] > SNAP_MAX_M:
        return None
    return {"x": float(pts[i, 0]), "y": 0.0, "z": float(pts[i, 1])}


def is_framed(bearing: float, distance_m: float, up_now: bool) -> bool:
    """Centred, with the camera tilt the distance calls for (either tilt in the hysteresis band)."""
    return abs(bearing) <= config.ATTENTION_CENTER_TOL_DEG and tilt_wanted(distance_m, up_now) == up_now


# --- the manager ----------------------------------------------------------------------------------


@dataclass
class Sighting:
    bearing_deg: float  # positive = right of the robot's heading
    distance_m: float
    xz: Tuple[float, float]  # contract frame


class RobotAttentionManager:
    """Keeps one person framed in a background thread; see the module docstring.

    `robot` gives: bridge (pose, publish_twist), camera (latest_pair), tilt (tilt_deg, set),
    nav (set_goal, status, cancel), moves (move, rotate), objects (update, track, people) and
    people (the PersonDetector). `clock` is injectable."""

    def __init__(self, robot: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._robot = robot
        self._clock = clock
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._pending_reason: Optional[str] = None
        self._reset_locked()

    def _reset_locked(self) -> None:
        self._state = "off"
        self._target: Optional[str] = None
        self._track_id: Optional[str] = None
        self._label: Optional[str] = None
        self._reason: Optional[str] = None
        self._since: Optional[float] = None
        self._ended_at: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._seen_since: Optional[float] = None
        self._seen_now = False
        self._distance: Optional[float] = None
        self._bearing: Optional[float] = None
        self._last_xz: Optional[Tuple[float, float]] = None
        self._moving = False
        self._history: Deque[Tuple[float, float, float]] = deque()
        self._drive_goal_id: Optional[str] = None
        self._drive_from_xz: Optional[Tuple[float, float]] = None
        self._replans = 0
        self._approach_block_until = 0.0
        self._last_backup = -1e9
        self._tilt_up = False
        self._standoff = config.PERSON_STANDOFF_M
        self._follow = False
        self._greet = False
        self._greeted = False

    # --- public API (the simulator's) ---

    def start(
        self, destination_id: str, standoff_m: Optional[float] = None, follow: bool = False, greet: bool = False
    ) -> Dict[str, Any]:
        if standoff_m is not None and not STANDOFF_RANGE_M[0] <= float(standoff_m) <= STANDOFF_RANGE_M[1]:
            raise AttentionError(f"standoff_m must be between {STANDOFF_RANGE_M[0]} and {STANDOFF_RANGE_M[1]} m.")
        people = self._robot.objects.people(max_age_s=config.PERSON_STALE_S * 3)
        if not people:
            raise AttentionError("No person in the scene to attend to.")
        track_id = destination_id[4:] if destination_id.startswith("obj:") else destination_id
        track = next((t for t in people if t.track_id == track_id), None)
        if track is None:
            pose = self._robot.bridge.pose()
            nearest = min(people, key=lambda t: t.distance_m if t.distance_m is not None else 1e9) if pose is None else min(
                people, key=lambda t: math.hypot(t.position["x"] - pose.x, t.position["z"] - pose.z)
            )
            raise AttentionError(
                f"Attention targets people: the person is 'obj:{nearest.track_id}'. Use set_nav_goal for places."
            )
        with self._state_lock:
            same = self._state in ACTIVE_STATES and self._target == destination_id
            if same:
                if standoff_m is not None:
                    self._standoff = float(standoff_m)
                self._follow = bool(follow)
                if greet and not self._greet:
                    self._greet, self._greeted = True, False
        if same:
            logger.info("Attention already on {}: options updated", destination_id)
            return self.status()
        self.cancel(wait=True, reason="replaced")
        if self._robot.nav.cancel(wait=True):
            logger.info("Attention preempted the active drive")
        cancel = threading.Event()
        with self._state_lock:
            self._reset_locked()
            self._state = "aiming"
            self._target = destination_id
            self._track_id = track.track_id
            self._label = config.PERSON_LABEL
            self._since = self._clock()
            self._last_xz = (float(track.position["x"]), float(track.position["z"]))
            self._standoff = float(standoff_m) if standoff_m is not None else config.PERSON_STANDOFF_M
            self._follow = bool(follow)
            self._greet = bool(greet)
            self._cancel = cancel
            self._pending_reason = None
        self._thread = threading.Thread(target=self._run, args=(cancel,), name="attention", daemon=True)
        self._thread.start()
        logger.info(
            "Attention on {} (standoff {:.1f} m{}{})", destination_id, self._standoff,
            ", following" if follow else "", ", greeting" if greet else "",
        )
        return self.status()

    def cancel(self, wait: bool = True, reason: str = "canceled") -> bool:
        thread = self._thread
        if thread is not None and thread.is_alive():
            with self._state_lock:
                self._pending_reason = reason
            self._cancel.set()
            if wait:
                thread.join(15.0)
                if thread.is_alive():
                    logger.warning("Attention thread did not stop within 15 s")
            return True
        return False

    def active(self) -> bool:
        with self._state_lock:
            return self._state in ACTIVE_STATES

    def target_xz(self) -> Optional[Tuple[float, float]]:
        with self._state_lock:
            return self._last_xz if self._state in ACTIVE_STATES else None

    def exclusion(self) -> Optional[Tuple[float, float, float]]:
        """(x, z, radius) of the attended person, kept out of the passage detector's depth points."""
        xz = self.target_xz()
        return (xz[0], xz[1], 0.45) if xz is not None else None

    def status(self) -> Dict[str, Any]:
        now = self._clock()
        with self._state_lock:
            end = self._ended_at if self._ended_at is not None else now
            seen_s = end - self._seen_since if self._seen_since is not None and self._seen_now else 0.0
            reference = self._last_seen if self._last_seen is not None else self._since
            unseen_s = end - reference if reference is not None and not self._seen_now else 0.0
            out = {
                "state": self._state,
                "target": self._target,
                "label": self._label,
                "distance_m": round(self._distance, 2) if self._distance is not None else None,
                "bearing_deg": round(self._bearing, 1) if self._bearing is not None else None,
                "seen_s": round(max(0.0, seen_s), 1),
                "unseen_s": round(max(0.0, unseen_s), 1),
                "moving": bool(self._moving),
                "tilt_deg": None,
                "standoff_m": self._standoff,
                "follow": self._follow,
                "greeted": self._greeted if self._greet else None,
                "since_s": round(end - self._since, 1) if self._since is not None else 0.0,
                "target_xz": [round(self._last_xz[0], 2), round(self._last_xz[1], 2)] if self._last_xz else None,
                "reason": self._reason,
            }
        out["tilt_deg"] = round(float(self._robot.tilt.tilt_deg()), 1) + 0.0
        return out

    # --- the worker ---

    def _run(self, cancel: threading.Event) -> None:
        try:
            while not cancel.is_set():
                started = self._clock()
                self.tick(started)
                with self._state_lock:
                    if self._state not in ACTIVE_STATES:
                        return
                cancel.wait(max(0.0, config.ATTENTION_TICK_S - (self._clock() - started)))
            with self._state_lock:
                reason = self._pending_reason or "canceled"
            self._finish("off", reason)
        except Exception as exc:  # noqa: BLE001 - never die silently
            logger.exception("Attention crashed")
            self._finish("off", f"internal error: {exc}"[:120])

    def _finish(self, state: str, reason: str) -> None:
        with self._state_lock:
            drive = self._drive_goal_id
            tilt_up = self._tilt_up
            self._drive_goal_id = None
            self._state = state
            self._reason = reason
            self._ended_at = self._clock()
        self._robot.bridge.publish_twist()  # stop any aiming turn
        if drive is not None:
            status = self._robot.nav.status()
            if (status.get("goal") or {}).get("goal_id") == drive and status.get("state") in ("planning", "following"):
                self._robot.nav.cancel(wait=True)
        if state == "lost" and tilt_up:
            try:
                self._robot.tilt.set(0.0)  # nobody to look up at any more
            except Exception as exc:  # noqa: BLE001
                logger.info("Attention: could not level the camera ({})", exc)
        logger.info("Attention ended: {} ({})", state, reason)

    def _sight(self, now: float) -> Tuple[Optional[Sighting], Optional[Any]]:
        """Run the person detector on the latest frame and return the target's sighting (or None)
        and the robot's pose (None without TF)."""
        robot = self._robot
        pose = robot.bridge.pose()
        try:
            frame = robot.camera.latest_pair()
        except Exception:  # noqa: BLE001 - no frame: nothing seen this tick
            return None, pose
        if pose is None or frame.age_s > config.ATTENTION_TICK_S * 4:
            return None, pose
        tilt = robot.tilt.tilt_deg()
        detections = robot.people.detect(frame.rgb, frame.depth_m)
        tracks = robot.objects.update(detections, frame.depth_m, frame.intr, pose, pose.camera_position(), -tilt)
        with self._state_lock:
            track_id = self._track_id
        for track in tracks:
            if track.track_id == track_id and track.bbox is not None and track.distance_m is not None:
                u = (track.bbox[0] + track.bbox[2]) / 2.0
                bearing = math.degrees(math.atan((u - frame.intr.cx) / frame.intr.fx))
                return Sighting(bearing, float(track.distance_m), (track.position["x"], track.position["z"])), pose
        return None, pose

    def _turn(self, bearing: float) -> None:
        """Aim: an angular velocity toward the bearing (positive bearing = right = clockwise = negative
        yaw rate in ROS). The controller's 0.5 s command hold stops the turn when ticks stop."""
        rate = max(-config.ATTENTION_TURN_MAX_RADS, min(config.ATTENTION_TURN_MAX_RADS, -config.ATTENTION_TURN_GAIN * bearing))
        self._robot.bridge.publish_twist(wz=rate)

    def tick(self, now: float) -> None:
        """One step of the behaviour (the thread calls it every ATTENTION_TICK_S)."""
        robot = self._robot
        with self._state_lock:
            if self._state not in ACTIVE_STATES:
                return
            drive = self._drive_goal_id
        # 1. an approach drive of our own
        if drive is not None:
            status = robot.nav.status()
            goal = status.get("goal") or {}
            driving = goal.get("goal_id") == drive and status.get("state") in ("planning", "following")
            if driving:
                track = robot.objects.track(self._track_id) if self._track_id else None
                with self._state_lock:
                    origin = self._drive_from_xz
                    replans = self._replans
                    max_replans = FOLLOW_MAX_REPLANS if self._follow else config.ATTENTION_MAX_REPLANS
                moved = (
                    math.hypot(track.position["x"] - origin[0], track.position["z"] - origin[1])
                    if track is not None and origin is not None and now - track.last_seen < 2.0
                    else 0.0
                )
                if moved > config.ATTENTION_REPLAN_M and replans < max_replans:
                    logger.info("Attention: the person moved {:.2f} m during the approach; re-planning", moved)
                    robot.nav.cancel(wait=True)
                    with self._state_lock:
                        self._drive_goal_id = None
                        self._replans += 1
                else:
                    with self._state_lock:
                        self._last_seen = now  # the camera follows the path, not the person
                    self._set_state("approaching")
                    return
            else:
                with self._state_lock:
                    self._drive_goal_id = None
                    if status.get("state") == "failed":
                        self._approach_block_until = now + config.ATTENTION_APPROACH_RETRY_S

        # 2. sight
        seen, pose = self._sight(now)
        tilt_up = robot.tilt.tilt_deg() > 1.0
        with self._state_lock:
            self._tilt_up = tilt_up
            self._seen_now = seen is not None
            if seen is not None:
                if self._seen_since is None or self._last_seen is None or now - self._last_seen > config.ATTENTION_TICK_S * 3:
                    self._seen_since = now
                self._last_seen = now
                self._history.append((now, seen.xz[0], seen.xz[1]))
                while self._history and now - self._history[0][0] > config.ATTENTION_STILL_S + 1.0:
                    self._history.popleft()
                self._last_xz = (float(seen.xz[0]), float(seen.xz[1]))
                self._distance = float(seen.distance_m)
                self._bearing = float(seen.bearing_deg)
                speed = speed_mps(self._history, now)
                self._moving = speed is not None and speed > config.ATTENTION_STILL_MPS
            history = list(self._history)
            last_seen = self._last_seen if self._last_seen is not None else self._since
            last_xz = self._last_xz

        if seen is None:
            if last_seen is None or now - last_seen > config.ATTENTION_LOST_S:
                self._finish("lost", f"out of sight for {config.ATTENTION_LOST_S:g} s")
                return
            if last_xz is not None and pose is not None:
                bearing = bearing_deg(pose.xz, pose.yaw_deg, last_xz)
                if abs(bearing) > config.ATTENTION_CENTER_TOL_DEG:
                    self._turn(bearing)
            self._set_state("searching")
            return

        # 3. aim
        if abs(seen.bearing_deg) > config.ATTENTION_CENTER_TOL_DEG:
            self._turn(seen.bearing_deg)
        else:
            robot.bridge.publish_twist()

        # 4. tilt
        want_up = tilt_wanted(seen.distance_m, tilt_up)
        if want_up != tilt_up:
            try:
                robot.tilt.set(config.CAMERA_TILT_UP_DEG if want_up else 0.0)
                with self._state_lock:
                    self._tilt_up = want_up
            except Exception as exc:  # noqa: BLE001
                logger.info("Attention: tilt failed ({})", exc)

        # 5. standoff
        with self._state_lock:
            block_until = self._approach_block_until
            last_backup = self._last_backup
            standoff, follow = self._standoff, self._follow
        approach_from = standoff + (config.ATTENTION_APPROACH_FROM_M - config.PERSON_STANDOFF_M)
        too_close = max(TOO_CLOSE_FLOOR_M, standoff - (config.PERSON_STANDOFF_M - config.ATTENTION_TOO_CLOSE_M))
        if seen.distance_m > approach_from and now >= block_until and (follow or is_still(history, now)):
            if self._start_approach(seen, now, want_up):
                self._set_state("approaching")
                return
        elif seen.distance_m < too_close and now - last_backup >= config.ATTENTION_BACKUP_EVERY_S:
            with self._state_lock:
                self._last_backup = now
            message = robot.moves.move("MoveBack", config.MOVE_STEP_MAX_M)
            if not message.startswith("Successfully"):
                logger.info("Attention: could not back up ({})", message[:80])

        # 6. state
        framed = is_framed(seen.bearing_deg, seen.distance_m, want_up)
        self._set_state("framed" if framed else "aiming")

        # 7. greet: once framed, a little wiggle
        with self._state_lock:
            wiggle = framed and self._greet and not self._greeted
            if wiggle:
                self._greeted = True
        if wiggle:
            for action, degrees in WIGGLE:
                robot.moves.rotate(action, degrees)
                time.sleep(WIGGLE_PAUSE_S)
            logger.info("Attention: greeted {} (wiggle)", self._target)

    def _start_approach(self, seen: Sighting, now: float, tilt_up: bool) -> bool:
        robot = self._robot
        with self._state_lock:
            standoff = self._standoff
            target, label = self._target, self._label
        if tilt_up:
            try:
                robot.tilt.set(0.0)  # a drive levels the camera, as set_nav_goal does
                with self._state_lock:
                    self._tilt_up = False
            except Exception as exc:  # noqa: BLE001
                logger.info("Attention: could not level the camera ({})", exc)
        try:
            result = robot.nav.set_goal(
                target, {"x": seen.xz[0], "y": 0.0, "z": seen.xz[1]}, label or config.PERSON_LABEL, config.DEFAULT_SPEED_MPS, kind="person"
            )
        except ValueError as exc:
            logger.info("Attention: no path to the standoff ({}); staying put", str(exc)[:100])
            with self._state_lock:
                self._approach_block_until = now + config.ATTENTION_APPROACH_RETRY_S
            return False
        with self._state_lock:
            self._drive_goal_id = result["goal_id"]
            self._drive_from_xz = (float(seen.xz[0]), float(seen.xz[1]))
        logger.info("Attention: approaching the person to {} m (drive {})", standoff, result["goal_id"])
        return True

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            if self._state in ACTIVE_STATES and self._state != state:
                logger.debug("Attention: {} -> {}", self._state, state)
            if self._state in ACTIVE_STATES:
                self._state = state


# --- the stand-in while the attention tools are not served ----------------------------------------


class NullAttention:
    """Attention that is always off: the robot's attention tools are not served (GPTPET_MCP_ATTENTION)."""

    def __init__(self, tilt_source: Callable[[], float]) -> None:
        self._tilt = tilt_source

    def start(self, destination_id: str, standoff_m: Optional[float] = None, follow: bool = False, greet: bool = False) -> Dict[str, Any]:
        raise AttentionError("No person in the scene to attend to.")

    def cancel(self, wait: bool = True, reason: str = "canceled") -> bool:
        return False

    def active(self) -> bool:
        return False

    def target_xz(self) -> Optional[Tuple[float, float]]:
        return None

    def exclusion(self) -> Optional[Tuple[float, float, float]]:
        return None

    def status(self) -> Dict[str, Any]:
        return attention_off(self._tilt())
