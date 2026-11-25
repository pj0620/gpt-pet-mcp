"""Parity: the simulator server and the robot server, both live, list the same tools with the same
schemas and answer with the same shapes. Needs both URLs (GPTPET_MCP_SIM_URL, GPTPET_MCP_ROBOT_URL).

Run with: pytest regression -m parity
"""

from __future__ import annotations

import pytest

from gpt_pet_mcp import contract
from regression.support import call, image_blocks, json_diff, list_resources, list_tools, shape, shapes_agree, tool_json

pytestmark = [pytest.mark.parity, pytest.mark.timeout(300)]


async def test_robot_exposes_exactly_the_contract(sim_url, robot_url):
    sim_tools = await list_tools(sim_url)
    robot_tools = await list_tools(robot_url)
    assert set(robot_tools) == set(contract.served_tools())
    assert set(robot_tools) <= set(sim_tools), f"not in the simulator: {set(robot_tools) - set(sim_tools)}"
    assert not any(name.startswith("sim_") for name in robot_tools)


async def test_tool_schemas_match(sim_url, robot_url):
    sim_tools = await list_tools(sim_url)
    robot_tools = await list_tools(robot_url)
    mismatches = []
    for name, tool in sorted(robot_tools.items()):
        if tool != sim_tools.get(name):
            mismatches.append(f"--- {name}\n{json_diff(sim_tools.get(name), tool)}")
    assert not mismatches, "\n".join(mismatches)


async def test_resources_match(sim_url, robot_url):
    sim_resources = await list_resources(sim_url)
    robot_resources = await list_resources(robot_url)
    assert set(robot_resources) == set(contract.RESOURCES) <= set(sim_resources)
    for uri in robot_resources:
        assert robot_resources[uri] == sim_resources[uri], json_diff(sim_resources[uri], robot_resources[uri])


async def test_get_nav_status_has_the_same_shape(sim_url, robot_url):
    sim = tool_json(await call(sim_url, "get_nav_status"))
    robot = tool_json(await call(robot_url, "get_nav_status"))
    assert set(sim) == set(robot) == {"state", "goal", "progress", "agent", "collisions", "reason"}
    assert shapes_agree(shape(sim), shape(robot)), json_diff(shape(sim), shape(robot))
    assert robot["state"] in ("idle", "planning", "following", "succeeded", "canceled", "failed")


async def test_list_destinations_has_the_same_shape(sim_url, robot_url):
    sim = tool_json(await call(sim_url, "list_destinations"))
    robot = tool_json(await call(robot_url, "list_destinations"))
    assert set(sim) == set(robot) == {"destinations"}
    assert isinstance(robot["destinations"], list)
    if sim["destinations"] and robot["destinations"]:
        assert shape(sim["destinations"][0]) == shape(robot["destinations"][0])


async def test_get_map_is_one_png_on_both(sim_url, robot_url):
    for url in (sim_url, robot_url):
        result = await call(url, "get_map")
        images = image_blocks(result)
        assert len(images) == 1 and images[0][0] == "image/png", url
        assert images[0][1][:8] == b"\x89PNG\r\n\x1a\n"


async def test_unknown_destination_is_refused_the_same_way(sim_url, robot_url):
    from regression.support import call_error

    expected = "Unknown destination id 'psg:999'. Call get_current_view or list_destinations first."
    assert await call_error(sim_url, "set_nav_goal", {"destination_id": "psg:999"}) == expected
    assert await call_error(robot_url, "set_nav_goal", {"destination_id": "psg:999"}) == expected
