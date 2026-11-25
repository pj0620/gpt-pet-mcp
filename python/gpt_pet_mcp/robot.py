"""The robot behind the tools: the composition root and the tool bodies.

`Robot` wires the ROS bridge, the camera and tilt, the navigation manager, the manual moves, the
destination registry, the object tracker and detector, and the attention object. Each tool in
main.py is a thin wrapper around one method here; the bodies mirror ai2thor-mcp's main.py step
for step, so both servers answer the same way. Failures the brain should read are `RobotError`s.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image as PILImage

from gpt_pet_mcp import config, contract
from gpt_pet_mcp.annotate import annotate_view
from gpt_pet_mcp.attention import AttentionError, NullAttention, RobotAttentionManager
from gpt_pet_mcp.camera import CameraError, CameraFeed, KinectTilt, TiltError
from gpt_pet_mcp.describe import DestinationRegistry, describe_object, describe_passage
from gpt_pet_mcp.detector import ObjectDetector, PersonDetector, load_vocabulary
from gpt_pet_mcp.errors import RobotError
from gpt_pet_mcp.frames import dist_xz
from gpt_pet_mcp.maprender import MapOverlays
from gpt_pet_mcp.moves import Moves
from gpt_pet_mcp.nav import NavError, RobotNavManager
from gpt_pet_mcp.objects import ObjectTracker
from gpt_pet_mcp.perception import detect_passages
from gpt_pet_mcp.ros_bridge import RosBridge

HEALTH_LOG_S = 10.0
POSE_ERROR = (
    "The robot's pose is unknown: no map->base_link transform yet "
    "(is slam_toolbox running and the lidar publishing?)."
)


class Robot:
    def __init__(self) -> None:
        self.bridge = RosBridge()
        self.camera = CameraFeed()
        self.tilt = KinectTilt(self.bridge)
        self.nav = RobotNavManager(self.bridge)
        self.moves = Moves(self.bridge)
        self.registry = DestinationRegistry()
        self.objects = ObjectTracker()
        # with the attention tools served, people come from the pose detector (keypoints: depth at
        # the torso, posture later), so the object detector's vocabulary drops its person prompt
        self.people = PersonDetector() if contract.ATTENTION_ENABLED else PersonDetector(model_name="none")
        vocabulary = load_vocabulary(config.VOCABULARY_PATH)
        if self.people.enabled:
            vocabulary = {prompt: label for prompt, label in vocabulary.items() if label != config.PERSON_LABEL}
        self.detector = ObjectDetector(vocabulary=vocabulary)
        self.attention = RobotAttentionManager(self) if contract.ATTENTION_ENABLED else NullAttention(self.tilt.tilt_deg)
        self._started = False
        self._stopping = False
        self._health_thread: Optional[threading.Thread] = None

    # --- lifecycle ---

    def start(self, wait_s: float = config.ROS_WAIT_S) -> None:
        if self._started:
            return
        self.bridge.start()
        self.camera.attach(self.bridge)
        self.tilt.attach(self.bridge)
        self.nav.set_blocker_source(lambda: self.objects.blocker_name(self.bridge.pose(), 0.0))
        self.moves.set_blocker_source(lambda bearing: self.objects.blocker_name(self.bridge.pose(), bearing))
        self.detector.load_in_background()
        self.people.load_in_background()
        self._started = True
        self._health_thread = threading.Thread(target=self._health_loop, name="health", daemon=True)
        self._health_thread.start()
        if wait_s > 0:
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                health = self.health()
                if health["map"] and health["pose"] and all(health["nav2"].values()):
                    break
                time.sleep(0.5)
        logger.info("Robot ready: {}", self.health())

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            self.attention.cancel(reason="server shutdown")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.nav.cancel(wait=True)
        except Exception:  # noqa: BLE001
            pass
        self.bridge.stop()

    def health(self) -> Dict[str, Any]:
        health = self.bridge.health()
        health["camera"] = self.camera.health()
        health["tilt"] = {"node": self.tilt.ready(), "commanded_deg": self.tilt.tilt_deg(), "measured_deg": self.tilt.measured()}
        health["detector"] = self.detector.health()
        health["people"] = self.people.health()
        health["attention"] = contract.ATTENTION_ENABLED
        health["nav_state"] = self.nav.status()["state"]
        return health

    def _all_green(self, health: Dict[str, Any]) -> bool:
        camera = health["camera"]
        return bool(
            health["map"]
            and health["pose"]
            and all(health["nav2"].values())
            and camera["rgb_age_s"] is not None
            and camera["rgb_age_s"] < 2.0
            and camera["depth_age_s"] is not None
            and health["tilt"]["node"]
            and (not self.detector.enabled or self.detector.loaded)
            and (not self.people.enabled or self.people.loaded)
        )

    def _health_loop(self) -> None:
        last_green: Optional[bool] = None
        while not self._stopping:
            try:
                health = self.health()
                green = self._all_green(health)
                if green != last_green or not green:
                    compact = {k: v for k, v in health.items() if k not in ("camera", "detector", "people", "tilt")}
                    compact["camera"] = {k: health["camera"][k] for k in ("rgb_age_s", "depth_age_s", "intrinsics")}
                    compact["tilt_node"] = health["tilt"]["node"]
                    compact["detector"] = health["detector"]["loaded"] if self.detector.enabled else "off"
                    compact["people"] = health["people"]["loaded"] if self.people.enabled else "off"
                    for key in ("detector", "people"):
                        if health[key]["error"]:
                            compact[f"{key}_error"] = health[key]["error"]
                    logger.info("Health {}: {}", "ok" if green else "degraded", compact)
                last_green = green
            except Exception:  # noqa: BLE001
                logger.exception("health check failed")
            time.sleep(HEALTH_LOG_S)

    # --- helpers ---

    def _pose(self):
        pose = self.bridge.pose()
        if pose is None:
            raise RobotError(POSE_ERROR)
        return pose

    def _camera_block(self) -> Dict[str, float]:
        return {"height_m": round(config.CAMERA_HEIGHT_M, 2) + 0.0, "tilt_deg": round(self.tilt.tilt_deg(), 1) + 0.0}

    def _free_cells_near(self, pose) -> Optional[np.ndarray]:
        cells = self.bridge.costmap.free_cells_xz()
        if cells is None:
            return None
        near = np.hypot(cells[:, 0] - pose.x, cells[:, 1] - pose.z) <= config.COSTMAP_RADIUS_M
        return cells[near]

    # --- get_current_view ---

    def current_view(self) -> Tuple[PILImage.Image, Dict[str, Any]]:
        try:
            frame = self.camera.latest_pair()
        except CameraError as exc:
            raise RobotError(str(exc)) from None
        pose = self._pose()
        tilt = self.tilt.tilt_deg()
        horizon = -tilt  # the simulator's cameraHorizon: positive pitches down
        camera_pos = pose.camera_position()
        now = time.monotonic()

        detections = self.detector.detect(frame.rgb) if self.detector.enabled else []
        if self.people.enabled:
            detections = detections + self.people.detect(frame.rgb, frame.depth_m)
        tracks = self.objects.update(detections, frame.depth_m, frame.intr, pose, camera_pos, horizon)
        object_records = []
        for track in tracks:
            if track.label in config.EXCLUDED_OBJECT_TYPES or track.bbox is None:
                continue
            bbox = track.bbox
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.MIN_OBJECT_BBOX_AREA_PX:
                continue
            obj = self.objects.obj_dict(track)
            description = describe_object(obj, pose.position, pose.yaw_deg)
            if track.is_person:
                speed = track.speed_mps(now)
                if speed is not None and speed > config.ATTENTION_STILL_MPS:
                    description += ", walking"
            object_records.append(self.registry.upsert_object(obj, description, bbox, obj["distance"]))
        object_records.sort(key=lambda r: r.distance_m or 0.0)
        object_records = object_records[: config.MAX_OBJECT_DESTINATIONS]

        passages = detect_passages(
            frame.depth_m,
            camera_pos,
            pose.position,
            pose.yaw_deg,
            horizon,
            frame.intr,
            reachable_xz=self._free_cells_near(pose),
            exclude_xz=self.attention.exclusion(),
        )
        passage_records = []
        for cand in passages:
            distance = dist_xz(pose.xz, (cand.waypoint["x"], cand.waypoint["z"]))
            passage_records.append(self.registry.upsert_passage(cand, describe_passage(cand), distance))

        for rec in self.registry.list():
            rec.marker = None
        view_records = object_records + passage_records
        for n, rec in enumerate(view_records, start=1):
            rec.marker = n

        attention = self.attention.status()
        annotated = annotate_view(frame.rgb, view_records, attention=attention)

        destinations: List[Dict[str, Any]] = []
        for rec in view_records:
            entry: Dict[str, Any] = {
                "marker": rec.marker,
                "id": rec.id,
                "kind": rec.kind,
                "label": rec.label,
                "description": rec.description,
            }
            if rec.distance_m is not None:
                entry["distance_m"] = round(rec.distance_m, 1)
            if rec.kind == "object" and rec.bbox is not None:
                entry["bbox"] = list(rec.bbox)
            if rec.kind == "passage" and rec.region_px is not None:
                entry["region"] = list(rec.region_px)
            destinations.append(entry)

        payload = {
            "agent": pose.payload(),
            "camera": self._camera_block(),
            "attention": attention,
            "destinations": destinations,
            "note": "Pass a destination id to set_nav_goal to drive there.",
        }
        logger.info("View: {} object and {} passage destinations", len(object_records), len(passage_records))
        return annotated, payload

    def rgb_png(self) -> Optional[bytes]:
        return self.camera.latest_rgb_png()

    # --- navigation ---

    def set_nav_goal(self, destination_id: str, speed_mps: float) -> dict:
        speed = float(min(max(speed_mps, config.MIN_SPEED_MPS), config.MAX_SPEED_MPS))
        ended_attention = self.attention.cancel(reason="a drive started")
        if self.tilt.tilt_deg() != 0.0:
            # a drive levels the camera (a tilt left behind would starve passage detection)
            logger.info("Levelling the camera for the drive")
            try:
                self.tilt.set(0.0)
            except TiltError as exc:
                logger.warning("could not level the camera: {}", exc)
        try:
            rec = self.registry.resolve(destination_id)
        except KeyError:
            raise RobotError(
                f"Unknown destination id '{destination_id}'. Call get_current_view or list_destinations first."
            ) from None

        target = rec.target
        kind = rec.kind
        if rec.kind == "object" and rec.object_id is not None:
            track = self.objects.track(rec.object_id)
            if track is not None:
                target = track.position
                if track.is_person:
                    age = time.monotonic() - track.last_seen
                    if age > config.PERSON_STALE_S:
                        raise RobotError(
                            f"Destination '{destination_id}' is stale: the person was last seen {age:.0f} s ago. "
                            "Call get_current_view for fresh destinations."
                        )
                    kind = "person"
        try:
            result = self.nav.set_goal(destination_id, target, rec.label, speed, kind)
        except NavError as exc:
            raise RobotError(f"Could not plan a path to '{destination_id}': {exc}") from None
        if ended_attention:
            result["ended_attention"] = True
        return result

    def nav_status(self) -> dict:
        return self.nav.status()

    def cancel_nav(self) -> dict:
        canceled = self.nav.cancel(wait=True)
        return {"canceled": canceled, "state": self.nav.status()["state"]}

    def list_destinations(self) -> dict:
        return {
            "destinations": [
                {
                    "id": rec.id,
                    "kind": rec.kind,
                    "label": rec.label,
                    "description": rec.description,
                    "target": [round(rec.target["x"], 2), round(rec.target["z"], 2)],
                }
                for rec in self.registry.list()
            ]
        }

    # --- map ---

    def map_image(self) -> PILImage.Image:
        pose = self.bridge.pose()
        attention_xz = self.attention.target_xz()
        overlays = MapOverlays(
            agent_position=pose.position if pose else None,
            agent_yaw_deg=pose.yaw_deg if pose else None,
            path=self.nav.path_snapshot(),
            destinations=self.registry.list(),
            goal_target=self.nav.goal_target(),
            attention_target={"x": attention_xz[0], "z": attention_xz[1]} if attention_xz else None,
        )
        return self.bridge.map.render(overlays)

    # --- manual control ---

    def do_move(self, action: str, magnitude: float) -> str:
        preempted = self.nav.preempt_for_manual()
        ended_attention = self.attention.cancel(reason="manual control")
        logger.info("Executing move action {} with magnitude {}", action, magnitude)
        message = self.moves.move(action, float(magnitude))
        if preempted:
            message += " (preempted active navigation)"
        if ended_attention:
            message += " (ended attention)"
        return message

    def do_rotate(self, action: str, degrees: float) -> str:
        preempted = self.nav.preempt_for_manual()
        ended_attention = self.attention.cancel(reason="manual control")
        logger.info("Executing rotate action {} with degrees = {}", action, degrees)
        message = self.moves.rotate(action, float(degrees))
        if preempted:
            message += " (preempted active navigation)"
        if ended_attention:
            message += " (ended attention)"
        return message

    def do_look(self, direction: str) -> str:
        up_deg = config.CAMERA_TILT_UP_DEG if direction == "up" else 0.0
        ended_attention = self.attention.cancel(reason="manual control")
        logger.info("Executing look {} (camera tilt {} deg)", direction, up_deg)
        try:
            self.tilt.set(up_deg)
        except TiltError as exc:
            return f"Look failed: {exc}"
        camera = self._camera_block()
        suffix = " (ended attention)" if ended_attention else ""
        if camera["tilt_deg"] > 0:
            return f"Camera tilted up {camera['tilt_deg']:g} degrees (height {camera['height_m']} m){suffix}"
        return f"Camera level (height {camera['height_m']} m){suffix}"

    # --- attention (Phase 4 serves these) ---

    def set_attention(self, destination_id: str, standoff_m: Optional[float], follow: bool, greet: bool) -> dict:
        try:
            return self.attention.start(destination_id, standoff_m, follow, greet)
        except AttentionError as exc:
            raise RobotError(str(exc)) from None

    def cancel_attention(self) -> dict:
        self.attention.cancel(reason="canceled")
        return self.attention.status()

    def attention_status(self) -> dict:
        return self.attention.status()
