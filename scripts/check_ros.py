#!/usr/bin/env python3
"""What the server would see on this machine's ROS 2 graph: imports, RMW, topics, Nav2 servers,
the robot's pose, the camera and the Kinect node. Run after `source env/pj-ubuntu.env`."""

from __future__ import annotations

import importlib
import sys
import time


def main() -> int:
    missing = []
    for name in ("rclpy", "nav2_msgs.action", "tf2_ros", "message_filters", "rcl_interfaces.srv", "fastmcp", "pydantic", "PIL", "numpy"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name}: {exc}")
    if missing:
        print("missing imports:\n  " + "\n  ".join(missing))
        print("source /opt/ros/humble/setup.bash (and env/pj-ubuntu.env) before running the server")
        return 1
    import rclpy

    from gpt_pet_mcp import config
    from gpt_pet_mcp.camera import CameraFeed, KinectTilt
    from gpt_pet_mcp.ros_bridge import RosBridge

    bridge = RosBridge("gpt_pet_mcp_check")
    bridge.start()
    camera = CameraFeed()
    camera.attach(bridge)
    tilt = KinectTilt(bridge)
    tilt.attach(bridge)
    print(f"rmw: {rclpy.utilities.get_rmw_implementation_identifier()}")
    time.sleep(3.0)
    topics = dict(bridge.node.get_topic_names_and_types())
    print(f"topics: {len(topics)} (publishers per topic the server reads; 0 = nobody publishes it yet)")
    for topic in (config.MAP_TOPIC, config.COSTMAP_TOPIC, config.SCAN_TOPIC, camera.rgb_topic, camera.depth_topic, config.CAMERA_INFO_TOPIC, config.TILT_STATE_TOPIC):
        publishers = bridge.node.count_publishers(topic)
        print(f"  {'ok  ' if publishers else 'none'} {topic} ({publishers} publisher{'s' if publishers != 1 else ''})")
    print(f"  {'ok  ' if bridge.node.count_subscribers(config.CMD_VEL_TOPIC) else 'none'} {config.CMD_VEL_TOPIC} ({bridge.node.count_subscribers(config.CMD_VEL_TOPIC)} subscribers: the drive controller)")
    health = bridge.health()
    print(f"nav2 servers: {health['nav2']}")
    print(f"map: {health['map']} (updates {health['map_updates']}), costmap: {health['costmap']}, scan: {health['scan']}")
    pose = bridge.pose()
    print(f"pose (contract frame): {pose.payload() if pose else None}")
    print(f"cmd_vel subscribers: {health['cmd_vel_subscribers']}")
    print(f"camera: {camera.health()}")
    print(f"kinect node parameter service: {tilt.ready()} ({config.KINECT_NODE}/set_parameters)")
    bridge.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
