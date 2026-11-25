"""Attention: the robot keeps a person framed on its own (gptpet-plans: People Interaction Design,
Attention Behaviour).

The pure helpers and the constants are ai2thor-mcp's `attention.py`, copied, so the robot's
behaviour will use the same geometry and thresholds. Until the person detector and the follower
exist on the robot (Phase 4), the tools see `NullAttention`: attention is always off and the three
attention tools are not served (contract.ATTENTION_ENABLED).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

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


# --- the stand-in until Phase 4 ----------------------------------------------------------------------


class NullAttention:
    """Attention that is always off: the robot has no person detector yet."""

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
