#!/usr/bin/env python3
"""Free space around the robot from the lidar, by sector of the robot's frame: what to check before
the robot tests move it. Run on pj-ubuntu after `source env/pj-ubuntu.env`.

    python scripts/clearance.py [--min 0.6]     # exit 1 when ahead/behind/left/right is closer than --min
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from gpt_pet_mcp import config
from gpt_pet_mcp.ros_bridge import RosBridge

SECTORS = {"ahead": 0.0, "left": 90.0, "behind": 180.0, "right": -90.0}  # robot frame, counter-clockwise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=float, default=0.6, help="metres free each way the tests move")
    parser.add_argument("--width", type=float, default=60.0, help="sector width, degrees")
    args = parser.parse_args()
    bridge = RosBridge("gpt_pet_mcp_clearance")
    bridge.start()
    deadline = time.monotonic() + 15.0
    scan = offset = None
    while time.monotonic() < deadline and (scan is None or offset is None):
        scan = bridge.scan(max_age_s=2.0)
        offset = bridge.frame_offset(config.BASE_FRAME, scan.frame_id) if scan is not None else None
        time.sleep(0.2)
    if scan is None:
        print("no /scan within 15 s")
        bridge.stop()
        return 2
    yaw = offset[2] if offset is not None else 0.0
    if offset is None:
        print(f"no TF {config.BASE_FRAME} <- {scan.frame_id}: assuming the lidar faces ahead")
    angles = np.degrees(scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment + yaw)
    ranges = scan.ranges
    valid = np.isfinite(ranges) & (ranges > max(scan.range_min, 0.12)) & (ranges < scan.range_max)
    tight = []
    for name, centre in SECTORS.items():
        rel = (angles - centre + 180.0) % 360.0 - 180.0
        mask = valid & (np.abs(rel) <= args.width / 2.0)
        nearest = float(np.min(ranges[mask])) if mask.any() else math.inf
        flag = "" if nearest >= args.min else "  <-- too close"
        if nearest < args.min:
            tight.append(name)
        print(f"{name:7s} {nearest:5.2f} m{flag}")
    pose = bridge.pose()
    print(f"pose: {pose.payload() if pose else None}")
    bridge.stop()
    return 1 if tight else 0


if __name__ == "__main__":
    raise SystemExit(main())
