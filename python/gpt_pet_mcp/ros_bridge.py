"""The one ROS 2 node behind the tools.

A `RosBridge` owns the rclpy node, spins it on a background `MultiThreadedExecutor` thread and
exposes what the tools need: the robot's pose from TF, the latest map, costmap, plan and scan,
`/cmd_vel` and `/speed_limit` publishers, Nav2 action clients and the Kinect node's parameter
service. The FastMCP tools run in worker threads and never spin rclpy themselves: every wait is a
`threading.Event` set from a future callback, or a cached value read under a lock. Subscription
callbacks only store what arrived (no decoding, no numpy work), so the executor never stalls.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from loguru import logger
from nav2_msgs.action import BackUp, ComputePathToPose, DriveOnHeading, NavigateToPose, Spin
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid, Path
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from gpt_pet_mcp import config
from gpt_pet_mcp.frames import Pose2D, yaw_of_quaternion
from gpt_pet_mcp.maprender import OccupancyGridRenderer

LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
)
"""slam_toolbox's /map and Nav2's costmaps are latched: a late subscriber still gets the last one."""
CMD_VEL_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
"""What cmd_vel_shaper and the teleop limiter publish with; the controller subscribes best-effort."""

STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "unknown",
    GoalStatus.STATUS_ACCEPTED: "accepted",
    GoalStatus.STATUS_EXECUTING: "executing",
    GoalStatus.STATUS_CANCELING: "canceling",
    GoalStatus.STATUS_SUCCEEDED: "succeeded",
    GoalStatus.STATUS_CANCELED: "canceled",
    GoalStatus.STATUS_ABORTED: "aborted",
}


def _stamp_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def now_s() -> float:
    """ROS time as float seconds (system clock, as the robot runs without sim time)."""
    return time.time()


class ActionCall:
    """One Nav2 action goal, driven from a worker thread.

    `start()` sends the goal; rclpy's futures complete on the executor thread and set the events
    here. `wait_accepted()` / `wait()` block the caller; `cancel()` asks Nav2 to stop and waits
    for the terminal result. `status` is a GoalStatus code (REJECTED when Nav2 refused the goal),
    `result` the action result message, `feedback` the last feedback message.
    """

    REJECTED = -1

    def __init__(self, client: ActionClient, goal: Any, name: str = "") -> None:
        self._client = client
        self._goal = goal
        self.name = name or client._action_name  # noqa: SLF001 - rclpy keeps the name private
        self._accepted = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self.handle: Any = None
        self.status: int = GoalStatus.STATUS_UNKNOWN
        self.result: Any = None
        self.feedback: Any = None
        self.error: Optional[str] = None
        self.started_at = time.monotonic()
        self.ended_at: Optional[float] = None

    def start(self) -> "ActionCall":
        future = self._client.send_goal_async(self._goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)
        return self

    def _on_feedback(self, msg: Any) -> None:
        with self._lock:
            self.feedback = msg.feedback

    def _on_goal_response(self, future: Any) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001 - the executor must never die on a client error
            self._fail(f"goal request failed: {exc}")
            return
        if handle is None or not handle.accepted:
            with self._lock:
                self.status = self.REJECTED
                self.ended_at = time.monotonic()
            self._accepted.set()
            self._done.set()
            return
        with self._lock:
            self.handle = handle
        self._accepted.set()
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future: Any) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"result failed: {exc}")
            return
        with self._lock:
            self.status = int(response.status)
            self.result = response.result
            self.ended_at = time.monotonic()
        self._done.set()

    def _fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.status = GoalStatus.STATUS_ABORTED
            self.ended_at = time.monotonic()
        self._accepted.set()
        self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def accepted(self) -> bool:
        return self._accepted.is_set() and self.status != self.REJECTED and self.error is None

    @property
    def status_name(self) -> str:
        return "rejected" if self.status == self.REJECTED else STATUS_NAMES.get(self.status, str(self.status))

    def wait_accepted(self, timeout_s: float) -> bool:
        return self._accepted.wait(timeout_s)

    def wait(self, timeout_s: Optional[float]) -> bool:
        return self._done.wait(timeout_s)

    def cancel(self, wait_s: float = config.CANCEL_TIMEOUT_S) -> bool:
        """Ask Nav2 to cancel; True when the goal then ended (any terminal state)."""
        if self.done:
            return True
        if not self._accepted.wait(1.0):
            return False
        handle = self.handle
        if handle is None:
            return self.done
        try:
            handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001
            logger.warning("{}: cancel request failed: {}", self.name, exc)
        return self._done.wait(wait_s)


@dataclass
class Scan:
    stamp_s: float
    frame_id: str
    angle_min: float
    angle_increment: float
    ranges: np.ndarray
    range_min: float
    range_max: float


@dataclass
class PlanSnapshot:
    stamp_s: float
    points_xy: np.ndarray  # (N, 2) in the map frame


class RosBridge:
    """See the module docstring. `start()` before use, `stop()` at shutdown."""

    def __init__(self, node_name: str = "gpt_pet_mcp") -> None:
        self._node_name = node_name
        self.node: Optional[Node] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self.map = OccupancyGridRenderer()
        self.costmap = OccupancyGridRenderer()
        self._map_stamp: Optional[float] = None
        self._costmap_stamp: Optional[float] = None
        self._plan: Optional[PlanSnapshot] = None
        self._scan: Optional[Scan] = None
        self._tf: Optional[Buffer] = None
        self._tf_listener: Optional[TransformListener] = None
        self._cmd_vel = None
        self._speed_limit = None
        self._param_clients: Dict[str, Any] = {}
        self.navigate: Optional[ActionClient] = None
        self.compute_path: Optional[ActionClient] = None
        self.spin: Optional[ActionClient] = None
        self.backup: Optional[ActionClient] = None
        self.drive: Optional[ActionClient] = None
        self._started_at: Optional[float] = None

    # --- lifecycle ---

    def start(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(self._node_name)
        node = self.node
        self._tf = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf, node, spin_thread=False)
        node.create_subscription(OccupancyGrid, config.MAP_TOPIC, self._on_map, LATCHED_QOS, callback_group=self._group)
        node.create_subscription(
            OccupancyGrid, config.COSTMAP_TOPIC, self._on_costmap, LATCHED_QOS, callback_group=self._group
        )
        node.create_subscription(Path, config.PLAN_TOPIC, self._on_plan, 10, callback_group=self._group)
        node.create_subscription(
            LaserScan, config.SCAN_TOPIC, self._on_scan, qos_profile_sensor_data, callback_group=self._group
        )
        self._cmd_vel = node.create_publisher(Twist, config.CMD_VEL_TOPIC, CMD_VEL_QOS)
        self._speed_limit = node.create_publisher(SpeedLimit, config.SPEED_LIMIT_TOPIC, 10)
        self.navigate = ActionClient(node, NavigateToPose, config.NAVIGATE_ACTION, callback_group=self._group)
        self.compute_path = ActionClient(node, ComputePathToPose, config.COMPUTE_PATH_ACTION, callback_group=self._group)
        self.spin = ActionClient(node, Spin, config.SPIN_ACTION, callback_group=self._group)
        self.backup = ActionClient(node, BackUp, config.BACKUP_ACTION, callback_group=self._group)
        self.drive = ActionClient(node, DriveOnHeading, config.DRIVE_ACTION, callback_group=self._group)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(node)
        self._thread = threading.Thread(target=self._spin, name="ros-executor", daemon=True)
        self._thread.start()
        self._started_at = time.monotonic()
        logger.info("ROS 2 node {} up (rmw {})", self._node_name, rclpy.utilities.get_rmw_implementation_identifier())

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:  # noqa: BLE001 - log, never die silently
            logger.exception("ROS executor stopped")

    def stop(self) -> None:
        try:
            self.stop_robot(5)
        except Exception:  # noqa: BLE001
            pass
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        logger.info("ROS 2 node {} stopped", self._node_name)

    @property
    def group(self) -> ReentrantCallbackGroup:
        return self._group

    # --- subscriptions ---

    def _on_map(self, msg: OccupancyGrid) -> None:
        info = msg.info
        self.map.update(info.width, info.height, info.resolution, info.origin.position.x, info.origin.position.y, msg.data)
        with self._lock:
            self._map_stamp = time.monotonic()

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        info = msg.info
        self.costmap.update(
            info.width, info.height, info.resolution, info.origin.position.x, info.origin.position.y, msg.data
        )
        with self._lock:
            self._costmap_stamp = time.monotonic()

    def _on_plan(self, msg: Path) -> None:
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float64).reshape(-1, 2)
        with self._lock:
            self._plan = PlanSnapshot(time.monotonic(), pts)

    def _on_scan(self, msg: LaserScan) -> None:
        scan = Scan(
            stamp_s=time.monotonic(),
            frame_id=msg.header.frame_id,
            angle_min=float(msg.angle_min),
            angle_increment=float(msg.angle_increment),
            ranges=np.asarray(msg.ranges, dtype=np.float32),
            range_min=float(msg.range_min),
            range_max=float(msg.range_max),
        )
        with self._lock:
            self._scan = scan

    def scan(self, max_age_s: float = 1.0) -> Optional[Scan]:
        with self._lock:
            scan = self._scan
        if scan is None or time.monotonic() - scan.stamp_s > max_age_s:
            return None
        return scan

    def plan(self, max_age_s: float = 30.0) -> Optional[np.ndarray]:
        with self._lock:
            plan = self._plan
        if plan is None or time.monotonic() - plan.stamp_s > max_age_s:
            return None
        return plan.points_xy

    def map_age_s(self) -> Optional[float]:
        with self._lock:
            stamp = self._map_stamp
        return None if stamp is None else time.monotonic() - stamp

    # --- TF ---

    def _lookup(self, target: str, source: str, timeout_s: float = config.TF_TIMEOUT_S):
        if self._tf is None:
            return None
        try:
            return self._tf.lookup_transform(target, source, Time(), timeout=Duration(seconds=timeout_s))
        except TransformException:
            return None
        except Exception as exc:  # noqa: BLE001 - a TF hiccup is a missing pose, not a crash
            logger.debug("TF lookup {} <- {} failed: {}", target, source, exc)
            return None

    def pose(self) -> Optional[Pose2D]:
        """The robot's pose in the map frame as the contract sees it, or None without TF."""
        t = self._lookup(config.MAP_FRAME, config.BASE_FRAME)
        if t is None:
            return None
        q = t.transform.rotation
        return Pose2D.from_ros(t.transform.translation.x, t.transform.translation.y, yaw_of_quaternion(q.x, q.y, q.z, q.w))

    def pose_age_s(self) -> Optional[float]:
        t = self._lookup(config.MAP_FRAME, config.BASE_FRAME, timeout_s=0.0)
        if t is None:
            return None
        return max(0.0, now_s() - _stamp_s(t.header.stamp))

    def odom_pose(self) -> Optional[Tuple[float, float, float]]:
        """(x, y, theta) of base_link in the odom frame: smooth, for measuring short moves."""
        t = self._lookup(config.ODOM_FRAME, config.BASE_FRAME)
        if t is None:
            return None
        q = t.transform.rotation
        return (t.transform.translation.x, t.transform.translation.y, yaw_of_quaternion(q.x, q.y, q.z, q.w))

    def frame_offset(self, target: str, source: str) -> Optional[Tuple[float, float, float]]:
        """(x, y, yaw) of `source` in `target`, for static frames such as the lidar's."""
        t = self._lookup(target, source)
        if t is None:
            return None
        q = t.transform.rotation
        return (t.transform.translation.x, t.transform.translation.y, yaw_of_quaternion(q.x, q.y, q.z, q.w))

    # --- publishers ---

    def cmd_vel_subscribers(self) -> int:
        return 0 if self._cmd_vel is None else int(self._cmd_vel.get_subscription_count())

    def publish_twist(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        if self._cmd_vel is None:
            return
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self._cmd_vel.publish(msg)

    def stop_robot(self, count: int = 5, period_s: float = 0.05) -> None:
        """Zero twists, as the teleop scripts send on exit."""
        for i in range(count):
            self.publish_twist()
            if i + 1 < count:
                time.sleep(period_s)

    def publish_speed_limit(self, speed_mps: float) -> None:
        if self._speed_limit is None:
            return
        msg = SpeedLimit()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.percentage = False
        msg.speed_limit = float(speed_mps)
        self._speed_limit.publish(msg)

    # --- actions ---

    def call(self, client: ActionClient, goal: Any, name: str = "") -> ActionCall:
        return ActionCall(client, goal, name).start()

    @staticmethod
    def ready(client: Optional[ActionClient]) -> bool:
        return client is not None and client.server_is_ready()

    def nav2_ready(self) -> Dict[str, bool]:
        return {
            config.NAVIGATE_ACTION: self.ready(self.navigate),
            config.COMPUTE_PATH_ACTION: self.ready(self.compute_path),
            config.SPIN_ACTION: self.ready(self.spin),
            config.BACKUP_ACTION: self.ready(self.backup),
            config.DRIVE_ACTION: self.ready(self.drive),
        }

    # --- parameters of other nodes (the Kinect's tilt) ---

    def set_parameter(self, node_name: str, name: str, value: float, timeout_s: float) -> Tuple[bool, str]:
        """Set a double parameter on another node. (ok, reason)."""
        service = f"{node_name.rstrip('/')}/set_parameters"
        client = self._param_clients.get(service)
        if client is None:
            client = self.node.create_client(SetParameters, service, callback_group=self._group)
            self._param_clients[service] = client
        if not client.service_is_ready():
            return False, f"{service} unavailable"
        request = SetParameters.Request()
        parameter = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))
        request.parameters = [parameter]
        done = threading.Event()
        box: Dict[str, Any] = {}

        def _on_done(future: Any) -> None:
            try:
                box["response"] = future.result()
            except Exception as exc:  # noqa: BLE001
                box["error"] = str(exc)
            done.set()

        client.call_async(request).add_done_callback(_on_done)
        if not done.wait(timeout_s):
            return False, f"{service} did not answer within {timeout_s:g} s"
        if "error" in box:
            return False, box["error"]
        results = list(box["response"].results)
        if not results:
            return False, "empty response"
        if not results[0].successful:
            return False, results[0].reason or "rejected"
        return True, ""

    def parameter_service_ready(self, node_name: str) -> bool:
        service = f"{node_name.rstrip('/')}/set_parameters"
        client = self._param_clients.get(service)
        if client is None:
            client = self.node.create_client(SetParameters, service, callback_group=self._group)
            self._param_clients[service] = client
        return bool(client.service_is_ready())

    # --- health ---

    def health(self) -> Dict[str, Any]:
        pose_age = self.pose_age_s()
        scan = self.scan(max_age_s=5.0)
        return {
            "ros": bool(rclpy.ok()),
            "domain": os.environ.get("ROS_DOMAIN_ID", "0"),
            "uptime_s": None if self._started_at is None else round(time.monotonic() - self._started_at, 1),
            "map": self.map.has_map,
            "map_updates": self.map.updates,
            "map_age_s": None if self.map_age_s() is None else round(self.map_age_s(), 1),
            "costmap": self.costmap.has_map,
            "pose": pose_age is not None,
            "pose_age_s": None if pose_age is None else round(pose_age, 2),
            "scan": scan is not None,
            "nav2": self.nav2_ready(),
            "cmd_vel_subscribers": self.cmd_vel_subscribers(),
        }
