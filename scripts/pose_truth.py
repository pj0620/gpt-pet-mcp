#!/usr/bin/env python3
"""The robot's pose from three sources at once, wall-clocked: wheel odometry
(/mecanum_drive_controller/odometry), the EKF (TF odom->base_link) and SLAM (TF map->base_link).
Run on pj-ubuntu after `source env/pj-ubuntu.env`; the campaign driver calls it before and after
each move so the contract's numbers can be checked against the robot's own.

    python scripts/pose_truth.py [--label before-rotate-90] [--settle 1.0]

Prints one JSON line: ROS-frame values (x, y, yaw_deg counter-clockwise) per source plus the
contract form of the SLAM pose (x, z, yaw clockwise), or null for a source that is not there.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import threading
import time

from nav_msgs.msg import Odometry

from gpt_pet_mcp import config
from gpt_pet_mcp.frames import yaw_of_quaternion, yaw_to_sim
from gpt_pet_mcp.ros_bridge import RosBridge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds to let the robot settle before reading")
    parser.add_argument("--wait", type=float, default=6.0, help="seconds to wait for TF and odometry")
    args = parser.parse_args()

    bridge = RosBridge("gpt_pet_mcp_pose_truth")
    bridge.start()
    wheel: dict = {}
    lock = threading.Lock()

    def on_odom(msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        with lock:
            wheel["x"] = msg.pose.pose.position.x
            wheel["y"] = msg.pose.pose.position.y
            wheel["yaw_deg"] = math.degrees(yaw_of_quaternion(q.x, q.y, q.z, q.w))

    bridge.node.create_subscription(Odometry, "/mecanum_drive_controller/odometry", on_odom, 10, callback_group=bridge.group)
    time.sleep(max(0.0, args.settle))
    deadline = time.monotonic() + args.wait
    ekf = slam = None
    while time.monotonic() < deadline:
        ekf = bridge.odom_pose()
        pose = bridge.pose()
        slam = pose
        with lock:
            have_wheel = bool(wheel)
        if ekf is not None and slam is not None and have_wheel:
            break
        time.sleep(0.2)
    with lock:
        wheel_out = dict(wheel) if wheel else None
    out = {
        "t": time.time(),
        "clock": dt.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "label": args.label,
        "wheel": wheel_out,
        "ekf": None if ekf is None else {"x": ekf[0], "y": ekf[1], "yaw_deg": math.degrees(ekf[2])},
        "slam": None if slam is None else {"x": slam.x, "y": slam.z, "yaw_deg": 90.0 - slam.yaw_deg},
        "contract": None if slam is None else slam.payload(),
    }
    bridge.stop()
    print(json.dumps(out))
    return 0 if slam is not None else 1


if __name__ == "__main__":
    sys.exit(main())
