#!/usr/bin/env python3
"""Render the live map the way get_map does and save it (plus a JSON snapshot of the grid), to look
at the robot's world from the server's point of view. Run after `source env/pj-ubuntu.env`.

    python scripts/capture_map.py [--out /tmp/map.png] [--json /tmp/map.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import time

from gpt_pet_mcp.maprender import MapOverlays
from gpt_pet_mcp.ros_bridge import RosBridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/gpt-pet-map.png")
    parser.add_argument("--json", default=None, help="also write the classified grid + pose as JSON")
    parser.add_argument("--wait", type=float, default=5.0, help="seconds to wait for /map")
    args = parser.parse_args()
    bridge = RosBridge("gpt_pet_mcp_capture")
    bridge.start()
    deadline = time.monotonic() + args.wait
    while not bridge.map.has_map and time.monotonic() < deadline:
        time.sleep(0.2)
    pose = bridge.pose()
    overlays = MapOverlays(agent_position=pose.position if pose else None, agent_yaw_deg=pose.yaw_deg if pose else None)
    image = bridge.map.render(overlays)
    image.save(args.out)
    print(f"map {'from /map' if bridge.map.has_map else 'placeholder (no /map yet)'} {image.size} -> {args.out}; pose {pose.payload() if pose else None}")
    if args.json:
        with bridge.map._lock:  # noqa: SLF001 - a debugging snapshot of the renderer's own state
            grid = bridge.map._image
            origin = bridge.map._origin
            res = bridge.map._resolution
        data = {
            "classified_png_palette": True,
            "h": None if grid is None else int(grid.shape[0]),
            "w": None if grid is None else int(grid.shape[1]),
            "res": res,
            "ox": origin[0],
            "oy": origin[1],
            "data": None if grid is None else base64.b64encode(grid.tobytes()).decode(),
            "robot": pose.payload() if pose else None,
        }
        with open(args.json, "w") as handle:
            json.dump(data, handle)
        print(f"grid -> {args.json}")
    bridge.stop()


if __name__ == "__main__":
    main()
