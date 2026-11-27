"""Views, destinations, the map and the camera tilt on the real robot (the Kinect on, SLAM running).

    GPTPET_MCP_ROBOT_OK=1 pytest regression/test_robot_view_live.py -m robot
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from gpt_pet_mcp import config
from regression.support import call, image_blocks, text_blocks, tool_json

pytestmark = [pytest.mark.robot, pytest.mark.timeout(300)]


async def test_view_has_the_frame_and_the_payload(robot_url):
    result = await call(robot_url, "get_current_view")
    images = image_blocks(result)
    assert len(images) == 1 and images[0][0] == "image/jpeg"
    img = Image.open(io.BytesIO(images[0][1]))
    assert img.width == config.FRAME_WIDTH_PX and img.height == round(config.FRAME_WIDTH_PX * 480 / 640)
    assert len(text_blocks(result)) == 1
    payload = tool_json(result)
    assert set(payload) == {"agent", "camera", "attention", "destinations", "note"}
    assert set(payload["agent"]) == {"x", "z", "yaw"}
    assert payload["camera"]["tilt_deg"] == 0.0 and 0.25 <= payload["camera"]["height_m"] <= 0.35
    assert payload["attention"]["state"] == "off"
    markers = [d["marker"] for d in payload["destinations"]]
    assert markers == list(range(1, len(markers) + 1))
    for dest in payload["destinations"]:
        assert dest["id"].startswith(("obj:", "psg:"))
        assert dest["kind"] in ("object", "passage")
        assert dest["description"].lower().startswith(dest["label"].lower()) or dest["kind"] == "passage"


async def test_list_destinations_remembers_the_view(robot_url):
    view = tool_json(await call(robot_url, "get_current_view"))
    known = tool_json(await call(robot_url, "list_destinations"))["destinations"]
    ids = {d["id"] for d in known}
    assert {d["id"] for d in view["destinations"]} <= ids
    for dest in known:
        assert len(dest["target"]) == 2


async def test_look_up_and_straight(robot_url):
    up = tool_json(await call(robot_url, "do_look", {"direction": "up"}))
    up = up["result"] if isinstance(up, dict) else up
    assert up.startswith("Camera tilted up 27 degrees"), up
    assert tool_json(await call(robot_url, "get_current_view"))["camera"]["tilt_deg"] == 27.0
    straight = tool_json(await call(robot_url, "do_look", {"direction": "straight"}))
    straight = straight["result"] if isinstance(straight, dict) else straight
    assert straight.startswith("Camera level"), straight
    assert tool_json(await call(robot_url, "get_current_view"))["camera"]["tilt_deg"] == 0.0


async def test_map_shows_the_robot(robot_url):
    images = image_blocks(await call(robot_url, "get_map"))
    img = Image.open(io.BytesIO(images[0][1])).convert("RGB")
    assert img.width > 100 and img.height > 100, "a real map, not the placeholder"
    reds = sum(count for count, colour in (img.getcolors(maxcolors=1 << 20) or []) if colour == config.AGENT_COLOR)
    assert reds > 0, "the robot triangle must be drawn"


async def test_rgb_resource_is_a_png(robot_url):
    from fastmcp import Client

    async with Client(robot_url) as client:
        contents = await client.read_resource("view://rgb")
    assert contents and contents[0].mimeType == "image/png"
