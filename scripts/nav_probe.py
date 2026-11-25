#!/usr/bin/env python3
"""Drive the real robot to a point a little ahead through RobotNavManager, in-process, and print
the status the brain would poll: the Phase 2 check of planning, following, progress and the
terminal state before destinations exist. Robot on clear floor; run after `source env/pj-ubuntu.env`.

    python scripts/nav_probe.py [--ahead 1.0] [--speed 0.35] [--timeout 120]
"""

from __future__ import annotations

import argparse
import json
import math
import time

from gpt_pet_mcp import config
from gpt_pet_mcp.nav import NavError, RobotNavManager
from gpt_pet_mcp.ros_bridge import RosBridge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ahead", type=float, default=1.0, help="metres ahead of the robot")
    parser.add_argument("--speed", type=float, default=config.DEFAULT_SPEED_MPS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--cancel-after", type=float, default=None, help="cancel the drive after this many seconds")
    args = parser.parse_args()
    bridge = RosBridge("gpt_pet_mcp_nav_probe")
    bridge.start()
    nav = RobotNavManager(bridge)
    deadline = time.monotonic() + 10.0
    pose = bridge.pose()
    while pose is None and time.monotonic() < deadline:
        time.sleep(0.2)
        pose = bridge.pose()
    if pose is None:
        print("no map->base_link pose: is slam_toolbox running?")
        bridge.stop()
        return 1
    fx, fz = pose.forward()
    target = {"x": pose.x + args.ahead * fx, "y": 0.0, "z": pose.z + args.ahead * fz}
    print(f"robot at {pose.payload()}, driving to ({target['x']:.2f}, {target['z']:.2f})")
    try:
        started = nav.set_goal("probe", target, "probe point", args.speed, kind="passage")
    except NavError as exc:
        print(f"could not start: {exc}")
        bridge.stop()
        return 1
    print(json.dumps(started))
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.timeout:
        status = nav.status()
        print(f"{time.monotonic() - t0:5.1f}s {status['state']:10s} {json.dumps(status['progress'])} agent={status['agent']} reason={status['reason']}")
        if status["state"] in ("succeeded", "failed", "canceled"):
            break
        if args.cancel_after is not None and time.monotonic() - t0 >= args.cancel_after:
            print("cancel ->", nav.cancel(wait=True))
            args.cancel_after = None
        time.sleep(0.5)
    final = bridge.pose()
    if final is not None:
        print(f"moved {math.hypot(final.x - pose.x, final.z - pose.z):.2f} m; final {final.payload()}")
    nav.cancel(wait=True)
    bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
