"""The robot contract: which of ai2thor-mcp's tools this server reproduces.

The simulator serves 33 tools; 12 of them plus the `view://rgb` resource are the robot contract
(gpt-pet-v2's profiles list them in `tool_filter`), the rest are simulator-only (`sim_*`, the
overhead recording). The base tools ship first; the attention tools arrive with the person
detector (Phase 4) and are registered only once ATTENTION_ENABLED is true, so the parity test
compares exactly what is served.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from gpt_pet_mcp import config

BASE_TOOLS = (
    "get_current_view",
    "set_nav_goal",
    "get_nav_status",
    "cancel_nav",
    "list_destinations",
    "get_map",
    "do_move",
    "do_rotate",
    "do_look",
)
ATTENTION_TOOLS = ("set_attention", "cancel_attention", "get_attention_status")
ROBOT_CONTRACT_TOOLS = BASE_TOOLS + ATTENTION_TOOLS
RESOURCES = ("view://rgb",)

ATTENTION_ENABLED = os.environ.get("GPTPET_MCP_ATTENTION", "0").lower() in ("1", "true", "yes")
"""GPTPET_MCP_ATTENTION=1 serves the attention tools (the person detector and the attention
behaviour, Phase 4). Off until verified on the robot; the brain's `[features] attention` follows."""


def served_tools() -> tuple:
    return ROBOT_CONTRACT_TOOLS if ATTENTION_ENABLED else BASE_TOOLS


def attention_off(tilt_deg: float = 0.0, reason: Optional[str] = None) -> Dict[str, Any]:
    """The attention status dict while nothing is attended, as ai2thor-mcp reports it."""
    return {
        "state": "off",
        "target": None,
        "label": None,
        "distance_m": None,
        "bearing_deg": None,
        "seen_s": 0.0,
        "unseen_s": 0.0,
        "moving": False,
        "tilt_deg": round(float(tilt_deg), 1) + 0.0,
        "standoff_m": config.PERSON_STANDOFF_M,
        "follow": False,
        "greeted": None,
        "since_s": 0.0,
        "target_xz": None,
        "reason": reason,
    }
