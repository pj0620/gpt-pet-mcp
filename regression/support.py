"""Helpers shared by the live tests: calling tools over MCP and reading their results the way the
brain does (gpt-pet-v2 mcp.py: image blocks and one JSON text block)."""

from __future__ import annotations

import base64
import difflib
import json
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError


def tool_json(result) -> Any:
    """The JSON payload of a call_tool result: the structured data when FastMCP returned one,
    else the (single) JSON text block."""
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    pytest.fail(f"expected one JSON text block, got {len(texts)}: {result.content!r}")


def image_blocks(result) -> List[Tuple[str, bytes]]:
    """(mime type, bytes) of every image block."""
    out = []
    for block in result.content:
        if getattr(block, "type", None) == "image":
            out.append((block.mimeType, base64.b64decode(block.data)))
    return out


def text_blocks(result) -> List[str]:
    return [block.text for block in result.content if getattr(block, "type", None) == "text"]


def shape(value: Any) -> Any:
    """The key structure and scalar types of a JSON value, None tolerated: what two servers must
    agree on even when the numbers differ."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def shapes_agree(a: Any, b: Any) -> bool:
    """Equal shapes, where None on either side matches anything (an optional field unset)."""
    if a is None or b is None:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(shapes_agree(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return not a or not b or shapes_agree(a[0], b[0])
    return a == b


def json_diff(a: Any, b: Any, left: str = "sim", right: str = "robot") -> str:
    return "\n".join(
        difflib.unified_diff(
            json.dumps(a, indent=2, sort_keys=True).splitlines(),
            json.dumps(b, indent=2, sort_keys=True).splitlines(),
            fromfile=left,
            tofile=right,
            lineterm="",
        )
    )


async def list_tools(url_or_server) -> Dict[str, dict]:
    async with Client(url_or_server) as client:
        tools = await client.list_tools()
    return {tool.name: tool.model_dump(mode="json", exclude_none=True) for tool in tools}


async def list_resources(url_or_server) -> Dict[str, dict]:
    async with Client(url_or_server) as client:
        resources = await client.list_resources()
    return {str(resource.uri): resource.model_dump(mode="json", exclude_none=True) for resource in resources}


async def call(url_or_server, tool: str, arguments: Optional[dict] = None, timeout_s: float = 120.0):
    async with Client(url_or_server, timeout=timeout_s) as client:
        return await client.call_tool(tool, arguments or {}, raise_on_error=True)


async def call_error(url_or_server, tool: str, arguments: Optional[dict] = None) -> str:
    """The tool error text a call raises; fails the test when the call succeeds."""
    try:
        await call(url_or_server, tool, arguments)
    except ToolError as exc:
        return str(exc)
    pytest.fail(f"{tool}({arguments}) was expected to fail")
