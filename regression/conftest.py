"""Shared fixtures. No mocks: the `parity`, `server` and `robot` tiers talk to the real servers
(the simulator on the Mac, this server on pj-ubuntu, the robot). Offline tests cover the geometry
and the copied modules, and the contract against a snapshot of the simulator's tool list."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

SIM_URL_ENV = "GPTPET_MCP_SIM_URL"
ROBOT_URL_ENV = "GPTPET_MCP_ROBOT_URL"
ROBOT_OK_ENV = "GPTPET_MCP_ROBOT_OK"
DEFAULT_SIM_URL = "http://127.0.0.1:8765/mcp"  # the Mac's simulator (ai2thor-mcp with FASTMCP_PORT=8765)
DEFAULT_ROBOT_URL = "http://pj-ubuntu.local:8000/mcp"  # gpt-pet-v2 real.ini


def mcp_alive(url: str, timeout_s: float = 5.0) -> bool:
    """The same health check gpt-pet-v2's simlaunch.mcp_alive makes: a JSON-RPC initialize."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "health", "version": "0"}},
        }
    ).encode()
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


@pytest.fixture(scope="session")
def sim_url() -> str:
    url = os.environ.get(SIM_URL_ENV, DEFAULT_SIM_URL)
    if not mcp_alive(url):
        pytest.fail(f"no simulator server answers at {url} (start ai2thor-mcp with FASTMCP_PORT or set {SIM_URL_ENV})")
    return url


@pytest.fixture(scope="session")
def robot_url() -> str:
    url = os.environ.get(ROBOT_URL_ENV, DEFAULT_ROBOT_URL)
    if not mcp_alive(url):
        pytest.fail(f"no robot server answers at {url} (run `gpt-pet-mcp --http` on pj-ubuntu or set {ROBOT_URL_ENV})")
    return url


@pytest.fixture(autouse=True)
def _robot_floor_clear(request: pytest.FixtureRequest) -> None:
    """Tests marked `robot` drive the real robot: the operator acknowledges a clear floor."""
    if request.node.get_closest_marker("robot") is None:
        return
    if os.environ.get(ROBOT_OK_ENV) != "1":
        pytest.fail(f"robot tests move the robot: put it on clear floor and set {ROBOT_OK_ENV}=1")


@pytest.fixture(scope="module")
def example_metadata() -> dict:
    """An AI2-THOR event's metadata, reduced to what the copied describe tests read (the
    simulator's data/example_metadata.json: FloorPlan2's spawn view, one visible counter top)."""
    return {
        "agent": {
            "name": "agent",
            "position": {"x": -1.0, "y": 0.9009996652603149, "z": 2.0},
            "rotation": {"x": 0.0, "y": 180.0, "z": 0.0},
            "cameraHorizon": -0.0,
        },
        "objects": [
            {
                "objectId": "CounterTop|-00.02|+00.96|+00.51",
                "objectType": "CounterTop",
                "position": {"x": -0.014999999664723873, "y": 0.9610999822616577, "z": 0.5059999823570251},
                "distance": 1.7904951572418213,
                "openable": False,
                "isOpen": False,
                "receptacle": True,
                "receptacleObjectIds": [
                    "Potato|+00.33|+00.94|+01.31",
                    "Bowl|+00.28|+00.92|+01.09",
                    "CellPhone|-00.31|+00.92|+01.29",
                    "Bread|+00.26|+00.99|-00.08",
                    "Egg|+00.13|+00.95|-00.17",
                    "Tomato|+00.17|+00.97|-00.28",
                    "Mug|-00.24|+00.92|-00.26",
                ],
                "toggleable": False,
                "isToggled": False,
                "isBroken": False,
                "visible": True,
            }
        ],
    }
