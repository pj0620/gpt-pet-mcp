"""Drives on the real robot through the MCP tools: short, on clear floor, every test ends stopped.

Run on a machine that reaches the robot server, with the robot on clear floor:
    GPTPET_MCP_ROBOT_OK=1 pytest regression/test_robot_drive_live.py -m robot
"""

from __future__ import annotations

import asyncio
import math
import re
import time

import pytest

from regression.support import call, tool_json

pytestmark = [pytest.mark.robot, pytest.mark.timeout(600)]

TERMINAL = {"succeeded", "failed", "canceled"}


async def _agent(url):
    status = tool_json(await call(url, "get_nav_status"))
    assert status["agent"] is not None, "the robot's pose is unknown (no map->base_link transform)"
    return status["agent"]


def _yaw_delta(before: float, after: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0


@pytest.fixture(autouse=True)
async def _stopped_after(robot_url):
    yield
    await call(robot_url, "cancel_nav")


async def test_rotate_left_turns_the_robot(robot_url):
    before = await _agent(robot_url)
    text = tool_json(await call(robot_url, "do_rotate", {"action": "RotateLeft", "degrees": 30}))
    assert text.startswith("Successfully rotated"), text
    await asyncio.sleep(1.0)
    after = await _agent(robot_url)
    # the contract's yaw grows clockwise: a left turn lowers it
    assert _yaw_delta(before["yaw"], after["yaw"]) == pytest.approx(-30.0, abs=8.0)


async def test_move_ahead_and_back(robot_url):
    before = await _agent(robot_url)
    ahead = tool_json(await call(robot_url, "do_move", {"action": "MoveAhead", "moveMagnitude": 0.2}))
    assert re.match(r"^(Successfully moved 0\.20 m ahead|Moved 0\.\d\d of 0\.20 m ahead, then blocked by .+|Move blocked by .+: did not move)", ahead), ahead
    await asyncio.sleep(1.0)
    mid = await _agent(robot_url)
    if ahead.startswith("Successfully"):
        assert math.hypot(mid["x"] - before["x"], mid["z"] - before["z"]) == pytest.approx(0.2, abs=0.08)
    back = tool_json(await call(robot_url, "do_move", {"action": "MoveBack", "moveMagnitude": 0.2}))
    assert back.startswith(("Successfully moved 0.20 m back", "Moved", "Move blocked")), back


async def test_strafe_left_reports_the_measured_distance(robot_url):
    text = tool_json(await call(robot_url, "do_move", {"action": "MoveLeft", "moveMagnitude": 0.2}))
    match = re.match(r"^(?:Successfully moved|Moved) (\d+\.\d\d)(?: of 0\.20)? m to the left", text)
    assert match or text.startswith("Move blocked by"), text
    if match:
        assert float(match.group(1)) >= 0.1


async def test_drive_to_the_nearest_destination_and_wait_it_out(robot_url):
    view = tool_json(await call(robot_url, "get_current_view"))
    candidates = sorted(
        (d for d in view["destinations"] if d.get("distance_m", 9.0) <= 2.0),
        key=lambda d: (d["kind"] != "passage", d.get("distance_m", 9.0)),
    )
    assert candidates, f"no destination within 2 m to drive to: {view['destinations']}"
    started = tool_json(await call(robot_url, "set_nav_goal", {"destination_id": candidates[0]["id"]}))
    assert started["state"] == "following" and started["path"]["length_m"] > 0
    goal_id = started["goal_id"]
    seen = []
    traveled = []
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        status = tool_json(await call(robot_url, "get_nav_status"))
        seen.append(status["state"])
        assert status["goal"]["goal_id"] == goal_id
        if status["progress"]:
            traveled.append(status["progress"]["distance_traveled_m"])
        if status["state"] in TERMINAL:
            break
        await asyncio.sleep(0.5)
    assert seen[-1] in TERMINAL, seen
    assert all(state in {"planning", "following"} for state in seen[:-1]), seen
    assert traveled == sorted(traveled), "distance traveled must not shrink"


async def test_cancel_stops_a_drive(robot_url):
    view = tool_json(await call(robot_url, "get_current_view"))
    candidates = [d for d in view["destinations"] if d.get("distance_m", 9.0) >= 1.0]
    assert candidates, f"no destination at least 1 m away: {view['destinations']}"
    started = tool_json(await call(robot_url, "set_nav_goal", {"destination_id": candidates[0]["id"]}))
    await asyncio.sleep(1.0)
    canceled = tool_json(await call(robot_url, "cancel_nav"))
    assert canceled == {"canceled": True, "state": "canceled"}, canceled
    status = tool_json(await call(robot_url, "get_nav_status"))
    assert status["state"] == "canceled" and status["goal"]["goal_id"] == started["goal_id"]
