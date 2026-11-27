"""Object destinations from detections: a map-frame position per box (its centre pixel through
the depth), a stable id per thing (`sofa#1`, `person#2`: a detection of the same label within
OBJECT_REID_M of a known one is that one), and the dicts `describe_object` and the registry take.

The simulator hands the brain AI2-THOR's object metadata; here the equivalent is built from what
the detector and the Kinect see, in the contract frame (x, z on the floor, y up).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from gpt_pet_mcp import config
from gpt_pet_mcp.frames import Pose2D, bearing_to, dist_xz
from gpt_pet_mcp.perception import rotation_world_from_camera


@dataclass
class Detection:
    label: str  # AI2-THOR style object type ("Sofa", "Person")
    conf: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    depth_m: Optional[float] = None  # a better depth than the box's centre (a person's hips and shoulders)


@dataclass
class Track:
    track_id: str  # "sofa#1"
    label: str
    position: Dict[str, float]  # contract frame {x, y, z}
    first_seen: float
    last_seen: float
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=40))  # (t, x, z)
    bbox: Optional[Tuple[int, int, int, int]] = None
    distance_m: Optional[float] = None
    conf: float = 0.0

    @property
    def is_person(self) -> bool:
        return self.label.lower() == config.PERSON_LABEL.lower()

    def speed_mps(self, now: float, window_s: float = 1.5) -> Optional[float]:
        recent = [s for s in self.history if now - s[0] <= window_s]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 0.3:
            return None
        (t0, x0, z0), (t1, x1, z1) = recent[0], recent[-1]
        return math.hypot(x1 - x0, z1 - z0) / (t1 - t0)


def box_depth_m(depth_m: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
    """Median planar depth over the central half of the box; None when the Kinect saw nothing there."""
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    cx1, cx2 = x1 + w // 4, max(x1 + w // 4 + 1, x2 - w // 4)
    cy1, cy2 = y1 + h // 4, max(y1 + h // 4 + 1, y2 - h // 4)
    cy1, cy2 = max(0, cy1), min(depth_m.shape[0], cy2)
    cx1, cx2 = max(0, cx1), min(depth_m.shape[1], cx2)
    if cy2 <= cy1 or cx2 <= cx1:
        return None
    patch = depth_m[cy1:cy2, cx1:cx2]
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    if valid.size < 4:
        return None
    return float(np.median(valid))


def pixel_to_world(u: float, v: float, depth: float, intr, camera_pos: Dict[str, float], yaw_deg: float, horizon_deg: float) -> Dict[str, float]:
    """One pixel at a planar depth -> contract-frame point (perception.depth_to_world's formula)."""
    x_c = depth * (u - intr.cx) / intr.fx
    y_c = -depth * (v - intr.cy) / intr.fy
    z_c = depth
    rot = rotation_world_from_camera(yaw_deg, horizon_deg)
    world = rot @ np.array([x_c, y_c, z_c]) + np.array([camera_pos["x"], camera_pos["y"], camera_pos["z"]])
    return {"x": float(world[0]), "y": float(world[1]), "z": float(world[2])}


class ObjectTracker:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._tracks: Dict[str, Track] = {}
        self._counters: Dict[str, int] = {}

    def update(
        self,
        detections: List[Detection],
        depth_m: np.ndarray,
        intr,
        pose: Pose2D,
        camera_pos: Dict[str, float],
        horizon_deg: float,
    ) -> List[Track]:
        """Place this view's detections in the world and match them to known tracks. Returns the
        tracks seen in this view, nearest first, each with its current bbox and distance."""
        now = self._clock()
        seen: List[Track] = []
        with self._lock:
            for det in detections:
                depth = det.depth_m if det.depth_m is not None else box_depth_m(depth_m, det.bbox)
                if depth is None:
                    continue
                u = (det.bbox[0] + det.bbox[2]) / 2.0
                v = (det.bbox[1] + det.bbox[3]) / 2.0
                position = pixel_to_world(u, v, depth, intr, camera_pos, pose.yaw_deg, horizon_deg)
                distance = dist_xz(pose.xz, (position["x"], position["z"]))
                track = self._match_locked(det.label, position)
                if track is None:
                    n = self._counters.get(det.label, 0) + 1
                    self._counters[det.label] = n
                    track = Track(f"{det.label.lower()}#{n}", det.label, dict(position), now, now)
                    self._tracks[track.track_id] = track
                track.position = dict(position)
                track.last_seen = now
                track.history.append((now, position["x"], position["z"]))
                track.bbox = tuple(int(v) for v in det.bbox)  # type: ignore[assignment]
                track.distance_m = distance
                track.conf = det.conf
                if track not in seen:
                    seen.append(track)
        seen.sort(key=lambda t: t.distance_m or 0.0)
        return seen

    def _match_locked(self, label: str, position: Dict[str, float]) -> Optional[Track]:
        radius = config.PERSON_REID_M if label.lower() == config.PERSON_LABEL.lower() else config.OBJECT_REID_M
        best, best_d = None, radius
        for track in self._tracks.values():
            if track.label != label:
                continue
            d = dist_xz((track.position["x"], track.position["z"]), (position["x"], position["z"]))
            if d <= best_d:
                best, best_d = track, d
        return best

    @staticmethod
    def obj_dict(track: Track) -> Dict[str, Any]:
        """What describe_object and DestinationRegistry.upsert_object read (AI2-THOR's keys)."""
        return {
            "objectId": track.track_id,
            "objectType": track.label,
            "position": dict(track.position),
            "distance": float(track.distance_m or 0.0),
        }

    def track(self, track_id: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(track_id)

    def tracks(self) -> List[Track]:
        with self._lock:
            return list(self._tracks.values())

    def people(self, max_age_s: Optional[float] = None) -> List[Track]:
        now = self._clock()
        with self._lock:
            return [t for t in self._tracks.values() if t.is_person and (max_age_s is None or now - t.last_seen <= max_age_s)]

    def blocker_name(
        self,
        pose: Optional[Pose2D],
        bearing_deg: float = 0.0,
        range_m: float = config.BLOCKER_RANGE_M,
        sector_deg: float = config.BLOCKER_SECTOR_DEG,
        max_age_s: float = 30.0,
    ) -> Optional[str]:
        """The label of the nearest known thing within `range_m` in a sector around `bearing_deg`
        (positive = right of the robot's heading): what a blocked move ran into, when known."""
        if pose is None:
            return None
        now = self._clock()
        best, best_d = None, range_m
        with self._lock:
            for track in self._tracks.values():
                if now - track.last_seen > max_age_s:
                    continue
                xz = (track.position["x"], track.position["z"])
                d = dist_xz(pose.xz, xz)
                if d > best_d:
                    continue
                rel = (bearing_to(pose, xz) - bearing_deg + 180.0) % 360.0 - 180.0
                if abs(rel) <= sector_deg / 2.0:
                    best, best_d = track, d
        return best.label if best is not None else None
