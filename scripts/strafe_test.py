#!/usr/bin/env python3
"""One strafe through the package's own Moves class, outside the server, so the strafe knobs can
be tried without a restart. Run on pj-ubuntu after `source env/pj-ubuntu.env`; the server must be
idle (both publish /cmd_vel).

    GPTPET_MCP_STRAFE_SPEED_MPS=0.12 GPTPET_MCP_STRAFE_YAW_KP=2.0 python scripts/strafe_test.py MoveLeft 0.3

Prints the tool text, the SLAM and odom (EKF) displacement in the robot's starting frame and the
heading change, like the campaign's pose truth.
"""

from __future__ import annotations

import math
import sys
import time

from gpt_pet_mcp import config
from gpt_pet_mcp.moves import Moves
from gpt_pet_mcp.ros_bridge import RosBridge


def rel(p0, p1):
    if p0 is None or p1 is None:
        return "n/a"
    th = p0[2]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dyaw = math.degrees((p1[2] - p0[2] + math.pi) % (2 * math.pi) - math.pi)
    return f"fwd {dx * math.cos(th) + dy * math.sin(th):+.3f} m  left {-dx * math.sin(th) + dy * math.cos(th):+.3f} m  dyaw {dyaw:+.1f} deg"


def slam_xyth(bridge: RosBridge):
    p = bridge.pose()
    if p is None:
        return None
    return (p.x, p.z, math.radians(90.0 - p.yaw_deg))


def main() -> int:
    action, magnitude = sys.argv[1], float(sys.argv[2])
    print(f"knobs: speed {config.STRAFE_SPEED_MPS} m/s, yaw kp {config.STRAFE_YAW_KP}, max wz {config.STRAFE_YAW_MAX_WZ}, abort {config.STRAFE_YAW_ABORT_DEG} deg")
    bridge = RosBridge("gpt_pet_mcp_strafe_test")
    bridge.start()
    deadline = time.monotonic() + 10.0
    while (bridge.odom_pose() is None or bridge.cmd_vel_subscribers() < 1 or slam_xyth(bridge) is None) and time.monotonic() < deadline:
        time.sleep(0.2)
    odom0, slam0 = bridge.odom_pose(), slam_xyth(bridge)
    t0 = time.time()
    text = Moves(bridge).move(action, magnitude)
    dt = time.time() - t0
    time.sleep(1.0)
    odom1, slam1 = bridge.odom_pose(), slam_xyth(bridge)
    bridge.stop()
    print(f"{action} {magnitude}: {text}  ({dt:.1f} s)")
    print(f"  odom/EKF  {rel(odom0, odom1)}")
    print(f"  SLAM      {rel(slam0, slam1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
