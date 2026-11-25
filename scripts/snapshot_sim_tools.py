#!/usr/bin/env python3
"""Snapshot the simulator's tool and resource list for regression/test_contract_offline.py.

Either from a running ai2thor-mcp server:
    python scripts/snapshot_sim_tools.py --url http://127.0.0.1:8765/mcp
or in-process, from the ai2thor-mcp checkout's environment (listing tools boots no Unity):
    cd ../ai2thor-mcp && uv run python ../gpt-pet-mcp/scripts/snapshot_sim_tools.py --import \
        --out ../gpt-pet-mcp/regression/data/sim_tools.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from fastmcp import Client

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "regression" / "data" / "sim_tools.json"


async def snapshot(target) -> dict:
    async with Client(target) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
    return {
        "tools": {tool.name: tool.model_dump(mode="json", exclude_none=True) for tool in sorted(tools, key=lambda t: t.name)},
        "resources": {str(r.uri): r.model_dump(mode="json", exclude_none=True) for r in resources},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="a running ai2thor-mcp server")
    parser.add_argument("--import", dest="in_process", action="store_true", help="import ai2thor_mcp.main in-process")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.in_process:
        from ai2thor_mcp import main as server

        target = server.mcp
    elif args.url:
        target = args.url
    else:
        parser.error("give --url or --import")
    data = asyncio.run(snapshot(target))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"{len(data['tools'])} tools, {len(data['resources'])} resources -> {args.out}")


if __name__ == "__main__":
    main()
