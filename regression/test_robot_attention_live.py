"""Attention on the real robot: a person takes part. Needs the server started with
GPTPET_MCP_ATTENTION=1 (the pose detector loaded) and the robot on clear floor.

Script for the person: stand about 2 m from the robot, ahead and a little to its left, and stay
still until the test prints "FRAMED"; then walk out of the camera's view and stay away. The tests
wait up to GPTPET_MCP_PERSON_WAIT_S (default 30) for someone to appear in the views.

    GPTPET_MCP_ROBOT_OK=1 pytest regression/test_robot_attention_live.py -m robot
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from gpt_pet_mcp import contract
from regression.support import call, call_error, tool_json

pytestmark = [pytest.mark.robot, pytest.mark.timeout(1800)]

PERSON_WAIT_S = float(os.environ.get("GPTPET_MCP_PERSON_WAIT_S", "30"))


async def _person_id(robot_url: str) -> str:
    deadline = time.monotonic() + PERSON_WAIT_S
    while time.monotonic() < deadline:
        view = tool_json(await call(robot_url, "get_current_view"))
        people = [d for d in view["destinations"] if d["label"] == "Person"]
        if people:
            return min(people, key=lambda d: d.get("distance_m", 9.0))["id"]
        await asyncio.sleep(1.0)
    pytest.fail(f"no Person destination in {PERSON_WAIT_S:g} s of views: is someone standing in front of the robot?")


async def test_attention_tools_are_served(robot_url):
    from regression.support import list_tools

    tools = await list_tools(robot_url)
    missing = set(contract.ATTENTION_TOOLS) - set(tools)
    assert not missing, f"start the server with GPTPET_MCP_ATTENTION=1 (missing {sorted(missing)})"


async def test_set_attention_refuses_a_place(robot_url):
    view = tool_json(await call(robot_url, "get_current_view"))
    places = [d for d in view["destinations"] if d["label"] != "Person"]
    if not places:
        pytest.fail("no non-person destination in view to test the refusal with")
    text = await call_error(robot_url, "set_attention", {"destination_id": places[0]["id"]})
    assert text.startswith("Attention targets people: the person is 'obj:") or text == "No person in the scene to attend to.", text


async def test_frames_a_still_person_then_loses_them(robot_url):
    person = await _person_id(robot_url)
    status = tool_json(await call(robot_url, "set_attention", {"destination_id": person}))
    assert status["state"] == "aiming" and status["target"] == person
    seen = []
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        status = tool_json(await call(robot_url, "get_attention_status"))
        seen.append(status["state"])
        if status["state"] == "framed":
            break
        await asyncio.sleep(0.5)
    assert status["state"] == "framed", seen
    assert status["bearing_deg"] is not None and abs(status["bearing_deg"]) <= 10.0
    assert status["distance_m"] is not None and 0.7 <= status["distance_m"] <= 3.5
    assert status["tilt_deg"] == 27.0, "closer than 2 m the camera tilts up"
    view = tool_json(await call(robot_url, "get_current_view"))
    assert view["attention"]["state"] in ("framed", "aiming") and view["attention"]["target"] == person
    print("FRAMED: the person now walks out of view and stays away")
    deadline = time.monotonic() + 40.0
    states = []
    while time.monotonic() < deadline:
        status = tool_json(await call(robot_url, "get_attention_status"))
        states.append(status["state"])
        if status["state"] == "lost":
            break
        await asyncio.sleep(0.5)
    assert status["state"] == "lost", states
    assert "searching" in states
    assert status["reason"] == "out of sight for 5 s"


async def test_manual_control_ends_attention(robot_url):
    person = await _person_id(robot_url)
    tool_json(await call(robot_url, "set_attention", {"destination_id": person}))
    await asyncio.sleep(1.0)
    text = tool_json(await call(robot_url, "do_rotate", {"action": "RotateLeft", "degrees": 15}))
    assert text.endswith("(ended attention)"), text
    status = tool_json(await call(robot_url, "get_attention_status"))
    assert status["state"] == "off" and status["reason"] == "manual control"
    await call(robot_url, "cancel_nav")
