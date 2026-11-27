"""The served tools against a snapshot of the simulator's tool list, without any server or ROS:
the robot server is imported in-process (its robot is created lazily, so nothing starts).

Refresh the snapshot from a running simulator with scripts/snapshot_sim_tools.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpt_pet_mcp import contract
from gpt_pet_mcp import main as server
from regression.support import json_diff, list_resources, list_tools

SNAPSHOT = Path(__file__).parent / "data" / "sim_tools.json"


@pytest.fixture(scope="module")
def sim_snapshot() -> dict:
    if not SNAPSHOT.exists():
        pytest.fail(f"{SNAPSHOT} is missing: run scripts/snapshot_sim_tools.py against the simulator")
    return json.loads(SNAPSHOT.read_text())


async def test_served_tools_are_exactly_the_contract():
    tools = await list_tools(server.mcp)
    assert set(tools) == set(contract.served_tools())
    assert not any(name.startswith("sim_") for name in tools)


async def test_tool_schemas_match_the_simulator(sim_snapshot):
    tools = await list_tools(server.mcp)
    sim_tools = sim_snapshot["tools"]
    missing = [name for name in tools if name not in sim_tools]
    assert not missing, f"not in the simulator: {missing}"
    mismatches = []
    for name, tool in sorted(tools.items()):
        if tool != sim_tools[name]:
            mismatches.append(f"--- {name}\n{json_diff(sim_tools[name], tool)}")
    assert not mismatches, "\n".join(mismatches)


async def test_resource_matches_the_simulator(sim_snapshot):
    resources = await list_resources(server.mcp)
    assert set(resources) == set(contract.RESOURCES)
    for uri, resource in resources.items():
        assert resource == sim_snapshot["resources"][uri], json_diff(sim_snapshot["resources"][uri], resource)
