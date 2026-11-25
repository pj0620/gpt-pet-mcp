"""The robot server against pj-ubuntu's live ROS 2 graph (slam_toolbox, Nav2). The robot itself
may be off: these tests check what the server answers either way, and that it answers several
clients at once.

Run with: pytest regression -m server   (GPTPET_MCP_ROBOT_URL, default http://pj-ubuntu.local:8000/mcp)
"""

from __future__ import annotations

import asyncio
import io
import time

import pytest
from fastmcp.exceptions import ToolError
from PIL import Image

from gpt_pet_mcp.maprender import FREE_WHITE, OCCUPIED_BLACK, UNKNOWN_GREY
from regression.support import call, call_error, image_blocks, text_blocks, tool_json

pytestmark = [pytest.mark.server, pytest.mark.timeout(300)]

NAV_STATES = {"idle", "planning", "following", "succeeded", "canceled", "failed"}
SOAK_S = 30.0
SOAK_CLIENTS = 5


async def test_nav_status_answers_without_a_goal(robot_url):
    status = tool_json(await call(robot_url, "get_nav_status"))
    assert status["state"] in NAV_STATES
    assert status["goal"] is None or set(status["goal"]) == {"goal_id", "destination_id", "label"}
    assert status["agent"] is None or set(status["agent"]) == {"x", "z", "yaw"}
    assert isinstance(status["collisions"], int)


async def test_unknown_destination_text_is_verbatim(robot_url):
    text = await call_error(robot_url, "set_nav_goal", {"destination_id": "psg:999"})
    assert text == "Unknown destination id 'psg:999'. Call get_current_view or list_destinations first."


async def test_map_is_a_png_in_the_ros_palette(robot_url):
    images = image_blocks(await call(robot_url, "get_map"))
    assert len(images) == 1 and images[0][0] == "image/png"
    img = Image.open(io.BytesIO(images[0][1])).convert("RGB")
    assert img.width >= 100 and img.height >= 100
    greys = {(v, v, v) for v in (UNKNOWN_GREY, FREE_WHITE, OCCUPIED_BLACK)}
    colours = img.getcolors(maxcolors=1 << 20) or []
    total = sum(count for count, _ in colours)
    palette = sum(count for count, colour in colours if colour in greys)
    assert palette / total > 0.9, "most pixels must be unknown/free/occupied greys"


async def test_list_destinations_is_a_list(robot_url):
    assert isinstance(tool_json(await call(robot_url, "list_destinations"))["destinations"], list)


async def test_view_works_or_says_why(robot_url):
    try:
        result = await call(robot_url, "get_current_view")
    except ToolError as exc:
        text = str(exc)
        assert text.startswith(("No camera frame", "No depth frame", "The robot's pose is unknown")), text
        return
    images = image_blocks(result)
    assert len(images) == 1 and images[0][0] == "image/jpeg"
    assert len(text_blocks(result)) == 1
    payload = tool_json(result)
    assert set(payload) == {"agent", "camera", "attention", "destinations", "note"}
    assert 0.25 <= payload["camera"]["height_m"] <= 0.35


async def test_look_works_or_says_why(robot_url):
    result = await call(robot_url, "do_look", {"direction": "straight"})
    text = tool_json(result)
    text = text["result"] if isinstance(text, dict) else text
    assert text.startswith(("Camera level", "Look failed: the Kinect node is not running")), text


async def test_many_clients_at_once(robot_url):
    """The portal, the brain, the trace sampler and the person watch all poll at once."""
    errors: list = []
    calls = {"n": 0}

    async def poll(tool: str) -> None:
        deadline = time.monotonic() + SOAK_S
        while time.monotonic() < deadline:
            try:
                await call(robot_url, tool, timeout_s=30.0)
                calls["n"] += 1
            except Exception as exc:  # noqa: BLE001 - collected, asserted below
                errors.append(f"{tool}: {exc!r}")
                return
            await asyncio.sleep(0.5)

    await asyncio.gather(*[poll("get_map" if i % 2 else "get_nav_status") for i in range(SOAK_CLIENTS)])
    assert not errors, errors[:5]
    # every call opens a stateless session (initialize + call) over WiFi: a cycle of a second or
    # two per client is the norm; what must not happen is a client starving while another renders
    assert calls["n"] >= SOAK_CLIENTS * SOAK_S / 3.0, f"only {calls['n']} calls in {SOAK_S} s"
