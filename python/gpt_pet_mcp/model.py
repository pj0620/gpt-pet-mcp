"""Wire shapes of the robot contract (copied from ai2thor-mcp model.py; documentation only)."""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

MOVE_ACTION = Literal["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]
ROTATE_ACTION = Literal["RotateRight", "RotateLeft"]
LOOK_DIRECTION = Literal["up", "straight"]

# --- small reusable shapes ---

class Vector3(TypedDict):
  x: float
  y: float
  z: float


class RGB(TypedDict):
  color: List[int]   # [r, g, b]
# --- navigation / destination wire shapes ---

NAV_STATE = Literal["idle", "planning", "following", "succeeded", "canceled", "failed"]

DESTINATION_KIND = Literal["object", "passage"]

PASSAGE_KIND = Literal["doorway", "hallway", "opening"]


class AgentPose2D(TypedDict):
  x: float
  z: float
  yaw: float


class NavGoalInfo(TypedDict):
  goal_id: str
  destination_id: str
  label: str


class NavProgress(TypedDict):
  distance_traveled_m: float
  distance_remaining_m: float
  percent: int
  eta_s: float
  waypoint: int
  waypoints_total: int


class NavStatus(TypedDict):
  state: NAV_STATE
  goal: Optional[NavGoalInfo]
  progress: Optional[NavProgress]
  agent: Optional[AgentPose2D]
  collisions: int
  reason: Optional[str]


class PlannedPath(TypedDict):
  waypoints: List[List[float]]  # [x, z] pairs
  length_m: float


class PlanSummary(TypedDict):
  goal_id: str
  destination_id: str
  label: str
  state: NAV_STATE
  path: PlannedPath
  speed_mps: float
  eta_s: float
  preempted_goal_id: Optional[str]


class DestinationPayload(TypedDict, total=False):
  marker: int
  id: str
  kind: DESTINATION_KIND
  label: str
  description: str
  distance_m: float
  bbox: List[int]    # objects: [x1, y1, x2, y2] in the current frame
  region: List[int]  # passages: [x1, y1, x2, y2] in the current frame
