"""Tunable constants and the switches read from the environment.

Two kinds live here. The perception, annotation, map and attention constants are the simulator's
(ai2thor-mcp `config.py`), so the modules copied from it (describe, annotate, perception,
maprender, attention helpers) behave the same on the robot. The `GPTPET_MCP_*` switches are this
server's own: where the camera sits, which ROS topics and nodes to use, the detector, the speed cap.
"""

import os


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- robot camera (a Kinect about a foot above the floor, with a motorised tilt) ---
CAMERA_HEIGHT_M = _env_float("GPTPET_MCP_CAMERA_HEIGHT", 0.3)
"""Height of the Kinect's optical centre above the floor. The brain's live test expects 0.25-0.35."""
CAMERA_FORWARD_M = _env_float("GPTPET_MCP_CAMERA_FORWARD", 0.08)
"""How far ahead of base_link the camera sits (metres). Measure once on the robot."""
CAMERA_TILT_UP_DEG = 27.0
"""How far do_look("up") tilts the camera above level: the Kinect's motor goes to +-30."""
FRAME_WIDTH_PX = int(_env_float("GPTPET_MCP_FRAME_WIDTH", 500))
"""Width of the frames the tools return. The brain converts a box's pixel column to a bearing
assuming a 500 px wide frame, so the Kinect's 640x480 is scaled to 500x375."""
CAMERA_TRANSPORT = _env_str("GPTPET_MCP_CAMERA_TRANSPORT", "compressed")
"""`compressed` subscribes to <topic>/compressed (JPEG) and <depth topic>/compressedDepth (PNG),
which the Jetson publishes with image_transport_plugins; `raw` takes the sensor_msgs/Image topics."""
RGB_TOPIC = _env_str("GPTPET_MCP_RGB_TOPIC", "/kinect/image_raw")
DEPTH_TOPIC = _env_str("GPTPET_MCP_DEPTH_TOPIC", "/kinect/depth/image_raw")
CAMERA_INFO_TOPIC = _env_str("GPTPET_MCP_CAMERA_INFO_TOPIC", "/kinect/camera_info")
KINECT_NODE = _env_str("GPTPET_MCP_KINECT_NODE", "/kinect/kinect_ros2")
"""The Kinect driver node; its `tilt_deg` parameter drives the tilt motor (set_parameters)."""
TILT_STATE_TOPIC = _env_str("GPTPET_MCP_TILT_STATE_TOPIC", "/kinect/tilt_state")
FRAME_MAX_AGE_S = 1.0  # a view needs a frame pair younger than this
FRAME_SYNC_SLOP_S = 0.1  # RGB and depth stamps this close count as one pair
CAMERA_INFO_FALLBACK = (525.0, 525.0, 319.5, 239.5)  # fx, fy, cx, cy of a Kinect v1 at 640x480
KINECT_WIDTH_PX = 640
KINECT_HEIGHT_PX = 480
TILT_TIMEOUT_S = 5.0

# --- object and person detection (pj-ubuntu's GPU) ---
DETECTOR_MODEL = _env_str("GPTPET_MCP_DETECTOR", "yolov8s-worldv2.pt")
"""Ultralytics YOLO-World weights for object destinations; "none" leaves only passages."""
POSE_MODEL = _env_str("GPTPET_MCP_POSE_MODEL", "yolov8n-pose.pt")
"""Ultralytics pose weights for the person detector (attention); "none" disables it."""
VOCABULARY_PATH = os.environ.get("GPTPET_MCP_VOCABULARY") or None
"""JSON mapping detector prompts to AI2-THOR object types; None uses the packaged vocabulary.json."""
DETECTOR_CONF = _env_float("GPTPET_MCP_DETECTOR_CONF", 0.3)
DETECTOR_DEVICE = _env_str("GPTPET_MCP_DETECTOR_DEVICE", "cuda:0")
OBJECT_REID_M = 0.5  # a detection this close to a known object of the same label is that object
PERSON_REID_M = 1.0  # people move: a wider match
PERSON_STALE_S = 10.0  # a person destination unseen this long cannot be driven to

# --- the ROS 2 graph (gpt-pet-v2/ros2_ws) ---
MAP_FRAME = "map"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
MAP_TOPIC = "/map"
COSTMAP_TOPIC = "/global_costmap/costmap"
PLAN_TOPIC = "/plan"
SCAN_TOPIC = "/scan"
CMD_VEL_TOPIC = "/cmd_vel"
SPEED_LIMIT_TOPIC = "/speed_limit"
NAVIGATE_ACTION = "navigate_to_pose"
COMPUTE_PATH_ACTION = "compute_path_to_pose"
SPIN_ACTION = "spin"
BACKUP_ACTION = "backup"
DRIVE_ACTION = "drive_on_heading"
ROS_WAIT_S = _env_float("GPTPET_MCP_ROS_WAIT_S", 0.0)
"""Block startup until the map, the pose and Nav2 are seen, at most this long; 0 starts at once
and the tools degrade (clear errors) until the graph is up."""
TF_TIMEOUT_S = 0.3
ROBOT_MAX_SPEED_MPS = _env_float("GPTPET_MCP_MAX_SPEED_MPS", 0.1)
"""What the robot drives at: Nav2's MPPI vx_max (nav2_params.yaml). set_nav_goal's speed_mps is
clamped to the contract's range but a drive never goes faster than this."""
BUMPER_FORWARD_M = 0.20  # the front bumper ahead of base_link (nav2's robot_radius)
LANE_HALF_WIDTH_M = 0.25
LANE_LOOK_M = 1.5  # how far ahead the Kinect lane check looks
LANE_MIN_HEIGHT_M = 0.06  # above the floor: below this is floor noise (the camera sits 0.3 m up, level)
LANE_MAX_HEIGHT_M = 0.6
LANE_MIN_POINTS = 60  # depth samples (2x2 subsampled) an obstacle must show before it counts
LANE_STOP_M = _env_float("GPTPET_MCP_LANE_STOP_M", 0.45)
"""do_move MoveAhead stops when the Kinect sees something in the lane closer than this to the
bumper. The lidar plane misses bottles, shoes and low boxes (2026-11-27: a bottle at 0.5 m was pushed
2 m with 'Successfully moved'); the depth sees them down to its ~0.5 m minimum range, so the stop
distance stays above that minus the camera-to-bumper offset. 0 disables the check."""
SIDESTEP_MODE = _env_str("GPTPET_MCP_SIDESTEP_MODE", "turn")
"""How MoveLeft/MoveRight are done: "turn" = turn 90 degrees to that side, drive the distance, turn
back (the mecanum wheels used like normal wheels; the user's choice 2026-11-27 after the strafes
slipped 2:1 sideways and yawed); "strafe" = a sideways velocity with the heading hold below."""
ROTATE_OVERSHOOT_DEG = _env_float("GPTPET_MCP_ROTATE_OVERSHOOT_DEG", 7.0)
"""Nav2's Spin stops this many degrees late on this robot (measured 2026-11-27: 10 -> 17, 30 -> 37,
90 -> 100); do_rotate asks for the requested angle minus this. 0 disables the correction."""
ROTATE_MIN_DEG = 3.0
STRAFE_SPEED_MPS = _env_float("GPTPET_MCP_STRAFE_SPEED_MPS", 0.07)
"""Lateral speed. The mecanum base slips sideways on hard floor and yaws while it does, so strafes
go slower than drives and hold their heading (below)."""
STRAFE_MAX_S = 15.0
STRAFE_CLEAR_M = 0.30  # a lidar return closer than this on the moving side stops a strafe
STRAFE_SECTOR_DEG = 60.0
STRAFE_YAW_KP = _env_float("GPTPET_MCP_STRAFE_YAW_KP", 2.0)
"""Heading hold during a strafe: wz = -KP * (yaw - yaw at start), from the odom (EKF) yaw. 0 turns
the hold off. Measured 2026-11-27 without it: a 0.3 m strafe yawed the robot 45-170 degrees."""
STRAFE_YAW_MAX_WZ = _env_float("GPTPET_MCP_STRAFE_YAW_MAX_WZ", 0.6)
STRAFE_YAW_ABORT_DEG = _env_float("GPTPET_MCP_STRAFE_YAW_ABORT_DEG", 35.0)
"""A strafe stops (reported as blocked by heading drift) when the hold cannot keep the heading
within this of the start."""
STRAFE_SLIP_RATIO = _env_float("GPTPET_MCP_STRAFE_SLIP_RATIO", 1.0)
"""Odometry sideways travel per metre the body really moves. Mecanum rollers slip sideways on hard
floor: 2026-11-27 the odometry read 0.34 m for 0.12-0.17 m of real lateral motion (camera parallax
on fixed objects), so a strafe aims for magnitude x this ratio by odometry and reports magnitude.
1.0 = trust the odometry."""
STRAFE_SQUARE_DEG = 3.0  # after a strafe the robot turns back on the spot to within this of its start heading
STRAFE_SQUARE_MAX_S = 4.0
PLAN_TIMEOUT_S = 5.0
CANCEL_TIMEOUT_S = 3.0
ACTION_ACCEPT_TIMEOUT_S = 5.0
OBJECT_STANDOFF_M = 0.5  # a drive to an object ends this far from its centre, facing it
STANDOFF_FALLBACKS_M = (0.7, 0.9)  # tried when no free floor is near the first standoff point
PROGRESS_SAMPLE_S = 0.1
PROGRESS_JUMP_CAP_M = 0.3  # a pose jump larger than this per sample is SLAM correcting, not driving
PATH_SIMPLIFY_M = 0.1  # Nav2 paths are dense; waypoints are the corners after this simplification
BLOCKER_SECTOR_DEG = 60.0
BLOCKER_RANGE_M = 1.0

# --- robot / navigation (the simulator's values that shape the contract) ---
ROBOT_WIDTH_M = 0.35
MIN_PASSAGE_WIDTH_M = 0.65
DEFAULT_SPEED_MPS = 0.35
MAX_SPEED_MPS = 1.0
MIN_SPEED_MPS = 0.05
GOAL_TOL_M = 0.30
MOVE_STEP_MAX_M = 0.25
BACKUP_M = 0.25

# --- passage detection (ai2thor-mcp perception.py) ---
DEPTH_STRIDE = 4
OBSTACLE_BAND_M = (0.05, 1.0)  # heights above floor that block the robot
FLOOR_Y_M = 0.0
FLOOR_TOL_M = _env_float("GPTPET_MCP_FLOOR_TOL", 0.06)
"""Depth points within this of the floor count as floor. Kinect depth noise and a tilt error of a
degree or two may want 0.10 on the robot (GPTPET_MCP_FLOOR_TOL)."""
MAX_RANGE_M = 5.0
MIN_FREE_DEPTH_M = 1.5
GAP_LEVELS_M = (1.5, 2.0, 2.5, 3.0, 4.0)  # level-set scan depths for gap finding
FLOOR_BEYOND_M = 0.3  # floor must be visible this far past the opening plane
FLOOR_SUPPORT_FRAC = 0.5  # ...in at least this fraction of the run's bins
AZ_BIN_DEG = 1.5
MIN_SECTOR_BINS = 2
FLANK_BINS = 2
GAP_BEYOND_M = 0.8  # opening must be at least this much deeper than its frame
OPEN_SPACE_MIN_M = 3.0  # flankless sectors must be clear at least this far
DOORWAY_MAX_WIDTH_M = 1.2
HALLWAY_MAX_WIDTH_M = 2.0
HALLWAY_MIN_DEPTH_M = 3.0
WAYPOINT_BEYOND_M = 0.5
SNAP_MAX_M = 0.35
PASSAGE_DEDUPE_M = 0.5
MAX_CEILING_Y_M = 2.2  # ignore points above this when boxing a passage region
COSTMAP_FREE_MAX = 40  # global costmap cells up to this cost count as floor the robot fits on
COSTMAP_RADIUS_M = 6.0  # free cells this far from the robot are offered for snapping

# --- person and attention (ai2thor-mcp attention.py) ---
PERSON_LABEL = "Person"
PERSON_STANDOFF_M = 1.3
"""How far from a person the robot stops: drives to the person (set_nav_goal) end here and
attention approaches to here."""
ATTENTION_TICK_S = 0.25
ATTENTION_CENTER_TOL_DEG = 10.0  # framed when the person is within this of the view's centre
ATTENTION_MAX_TURN_DEG = 45.0  # the largest turn per tick
ATTENTION_NEAR_M = 2.0  # closer than this the camera tilts up...
ATTENTION_LEVEL_M = 2.5  # ...farther than this it levels (either in between: hysteresis)
ATTENTION_APPROACH_FROM_M = 1.8  # a still person farther than this is approached to the standoff
ATTENTION_TOO_CLOSE_M = 0.9  # closer than this the robot backs up a step
ATTENTION_BACKUP_EVERY_S = 2.0
ATTENTION_STILL_MPS = 0.2  # slower than this counts as standing still...
ATTENTION_STILL_S = 2.0  # ...for this long before the robot approaches (no chasing)
ATTENTION_LOST_S = 5.0  # out of sight this long: attention ends ("lost")
ATTENTION_REPLAN_M = 0.5  # the person moved this far during an approach: re-plan
ATTENTION_MAX_REPLANS = 3
ATTENTION_APPROACH_RETRY_S = 5.0  # after a failed or impossible approach, wait this long
ATTENTION_TURN_GAIN = 0.02  # rad/s per degree of bearing while aiming
ATTENTION_TURN_MAX_RADS = 0.6
ATTENTION_COLOR = (255, 80, 220)  # the attended person's box, the banner, the map ring

# --- annotation / map (ai2thor-mcp) ---
MARKER_RADIUS_PX = 14
OBJECT_COLOR = (40, 200, 40)
PASSAGE_COLOR = (0, 190, 255)
GOAL_COLOR = (255, 60, 200)
PATH_COLOR = (255, 210, 40)
AGENT_COLOR = (230, 50, 50)
FONT_SIZE = 18
MIN_OBJECT_BBOX_AREA_PX = 400  # skip labels for sub-20x20px detections
MAX_OBJECT_DESTINATIONS = 20  # nearest N objects per view, to bound clutter
EXCLUDED_OBJECT_TYPES = {"Floor"}  # structural pseudo-objects, useless as destinations
