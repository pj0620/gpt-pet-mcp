"""The ROS <-> contract frame conversions, pinned so the copied geometry keeps working."""

import math

import pytest

from gpt_pet_mcp import frames
from gpt_pet_mcp.describe import direction_word, relative_bearing


def test_yaw_conversions_round_trip():
    assert frames.yaw_to_sim(0.0) == pytest.approx(90.0)  # ROS +x = contract +x = yaw 90
    assert frames.yaw_to_sim(math.pi / 2) == pytest.approx(0.0)  # ROS +y = contract +z = yaw 0
    assert frames.yaw_to_sim(math.pi) == pytest.approx(270.0)
    assert frames.yaw_to_sim(-math.pi / 2) == pytest.approx(180.0)
    for theta in (-3.0, -1.0, 0.0, 0.7, 2.5):
        back = frames.yaw_to_ros(frames.yaw_to_sim(theta))
        assert math.cos(back) == pytest.approx(math.cos(theta)) and math.sin(back) == pytest.approx(math.sin(theta))


def test_yaw_grows_clockwise_seen_from_above():
    # a counter-clockwise ROS turn (theta up) is a counter-clockwise contract turn (yaw down)
    assert frames.yaw_to_sim(0.3) < frames.yaw_to_sim(0.0)


def test_pose_from_ros_and_forward_vector():
    pose = frames.Pose2D.from_ros(1.5, -2.0, 0.0)  # facing ROS +x
    assert (pose.x, pose.z) == (1.5, -2.0)
    assert pose.yaw_deg == pytest.approx(90.0)
    fx, fz = pose.forward()
    assert (fx, fz) == pytest.approx((1.0, 0.0))  # forward = +x in both frames
    assert pose.payload() == {"x": 1.5, "z": -2.0, "yaw": 90.0}
    camera = pose.camera_position(height_m=0.3, forward_m=0.1)
    assert camera == pytest.approx({"x": 1.6, "y": 0.3, "z": -2.0})


def test_bearing_words_match_what_the_robot_sees():
    # robot at the origin facing ROS +x; a target at ROS (1, -1) is ahead and to its right
    pose = frames.Pose2D.from_ros(0.0, 0.0, 0.0)
    target = {"x": 1.0, "y": 0.0, "z": -1.0}  # contract (x, z) = ROS (x, y)
    bearing = relative_bearing(pose.position, pose.yaw_deg, target)
    assert bearing == pytest.approx(45.0)
    assert direction_word(bearing) == "ahead-right"
    assert frames.bearing_to(pose, (1.0, -1.0)) == pytest.approx(45.0)
    # ... and a target at ROS (0, 2) is to its left
    assert direction_word(relative_bearing(pose.position, pose.yaw_deg, {"x": 0.0, "z": 2.0})) == "left"


def test_heading_toward_faces_the_target():
    assert frames.heading_toward((0.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)  # +z
    assert frames.heading_toward((0.0, 0.0), (1.0, 0.0)) == pytest.approx(90.0)  # +x
    theta = frames.yaw_to_ros(frames.heading_toward((0.0, 0.0), (1.0, 0.0)))
    assert theta == pytest.approx(0.0)  # +x in ROS


def test_quaternion_yaw_round_trip():
    for theta in (-2.0, -0.5, 0.0, 1.0, 3.0):
        x, y, z, w = frames.quaternion_from_yaw(theta)
        assert frames.yaw_of_quaternion(x, y, z, w) == pytest.approx(theta)
