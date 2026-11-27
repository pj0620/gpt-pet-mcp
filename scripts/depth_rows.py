#!/usr/bin/env python3
"""What the Kinect's depth says about the camera itself: the depth of the top and bottom rows
(the tilt check: tilted up, the near floor leaves the bottom rows), a floor-plane fit of the lower
half of the frame (camera height above the floor and pitch), and the range to whatever is dead
ahead compared with the lidar's (the camera's forward offset from the lidar). Optionally saves the
raw frame as a fixture. Run on pj-ubuntu after `source env/pj-ubuntu.env`.

    python scripts/depth_rows.py [--rows 10] [--save regression/data/robot_frames/p5_level]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from gpt_pet_mcp import config
from gpt_pet_mcp.camera import CameraError, CameraFeed, KinectTilt
from gpt_pet_mcp.ros_bridge import RosBridge

LIDAR_FORWARD_M = 0.08  # base_link -> laser x offset in server.launch.py


def floor_plane(depth_m: np.ndarray, intr, rows_from: float = 0.55) -> dict:
    """Fit a plane to the lower part of the frame (camera coordinates: x right, y up, z forward).
    Returns the camera's height above that plane and its pitch (positive = looking down)."""
    h, w = depth_m.shape
    vs, us = np.mgrid[int(h * rows_from) : h : 2, 0:w:2]
    d = depth_m[vs, us]
    ok = np.isfinite(d) & (d > 0.3) & (d < 3.0)
    if ok.sum() < 200:
        return {"points": int(ok.sum()), "height_m": None, "pitch_deg": None}
    d, us, vs = d[ok], us[ok], vs[ok]
    pts = np.stack([d * (us - intr.cx) / intr.fx, -d * (vs - intr.cy) / intr.fy, d], axis=1)
    best = None
    rng = np.random.default_rng(0)
    for _ in range(60):  # a small RANSAC: furniture legs and boxes must not pull the floor
        sample = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        c = -float(n @ sample[0])
        dist = np.abs(pts @ n + c)
        inliers = dist < 0.03
        if best is None or inliers.sum() > best[0]:
            best = (int(inliers.sum()), inliers)
    inliers = best[1]
    centre = pts[inliers].mean(axis=0)
    _, _, vt = np.linalg.svd(pts[inliers] - centre)
    n = vt[-1]
    if n[1] < 0:
        n = -n  # the floor's normal points up
    c = -float(n @ centre)
    height = abs(c) / float(np.linalg.norm(n))
    pitch = math.degrees(math.atan2(n[2], n[1]))  # camera pitched down -> the normal tips toward +z
    return {"points": int(len(pts)), "inliers": int(inliers.sum()), "height_m": round(height, 3), "pitch_deg": round(pitch, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--save", default=None, help="path prefix for <prefix>.npz + <prefix>.json")
    parser.add_argument("--wait", type=float, default=8.0)
    args = parser.parse_args()
    bridge = RosBridge("gpt_pet_mcp_depth_rows")
    bridge.start()
    camera = CameraFeed()
    camera.attach(bridge)
    tilt = KinectTilt(bridge)
    tilt.attach(bridge)
    frame = None
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        try:
            frame = camera.latest_pair(max_age_s=2.0)
            break
        except CameraError:
            time.sleep(0.25)
    if frame is None:
        print(json.dumps({"error": "no Kinect frame"}))
        bridge.stop()
        return 1
    depth = frame.depth_m
    n = args.rows

    def rows_stat(block: np.ndarray) -> dict:
        valid = block[np.isfinite(block)]
        return {"median_m": round(float(np.median(valid)), 3) if valid.size else None, "invalid_frac": round(1.0 - valid.size / block.size, 3)}

    centre = depth[depth.shape[0] // 2 - 5 : depth.shape[0] // 2 + 5, depth.shape[1] // 2 - 5 : depth.shape[1] // 2 + 5]
    ahead_depth = float(np.nanmedian(centre)) if np.isfinite(centre).any() else None
    scan = bridge.scan(max_age_s=2.0)
    ahead_lidar = None
    if scan is not None:
        offset = bridge.frame_offset(config.BASE_FRAME, scan.frame_id)
        yaw = offset[2] if offset is not None else 0.0
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment + yaw
        rel = (np.degrees(angles) + 180.0) % 360.0 - 180.0
        sector = (np.abs(rel) <= 3.0) & np.isfinite(scan.ranges) & (scan.ranges > 0.12)
        if sector.any():
            ahead_lidar = float(np.median(scan.ranges[sector]))
    pose = bridge.pose()
    out = {
        "t": time.time(),
        "clock": dt.datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "tilt_commanded_deg": tilt.tilt_deg(),
        "tilt_measured_deg": tilt.measured(),
        "frame": {"width": int(depth.shape[1]), "height": int(depth.shape[0]), "age_s": round(frame.age_s, 2)},
        "top_rows": rows_stat(depth[:n]),
        "bottom_rows": rows_stat(depth[-n:]),
        "floor": floor_plane(depth, frame.intr),
        "ahead_depth_m": None if ahead_depth is None else round(ahead_depth, 3),
        "ahead_lidar_m": None if ahead_lidar is None else round(ahead_lidar, 3),
        "camera_forward_est_m": None if ahead_depth is None or ahead_lidar is None else round(LIDAR_FORWARD_M + ahead_lidar - ahead_depth, 3),
        "pose": pose.payload() if pose else None,
    }
    if args.save:
        path = Path(args.save).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path) + ".npz",
            rgb=frame.rgb,
            depth_m=depth,
            intr=np.array([frame.intr.fx, frame.intr.fy, frame.intr.cx, frame.intr.cy, frame.intr.width, frame.intr.height]),
            tilt_deg=np.array([tilt.tilt_deg()]),
        )
        Path(str(path) + ".json").write_text(json.dumps(out, indent=2) + "\n")
        out["saved"] = str(path) + ".npz"
    bridge.stop()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
