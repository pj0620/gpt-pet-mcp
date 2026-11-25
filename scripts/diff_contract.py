#!/usr/bin/env python3
"""Compare the contract tools' signatures and docstrings between ai2thor-mcp's main.py and this
server's main.py, from the source alone (no server, no ROS). Exit 1 on any difference.

    python scripts/diff_contract.py [--sim ../ai2thor-mcp/python/ai2thor_mcp/main.py]
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent.parent
ROBOT_MAIN = HERE / "python" / "gpt_pet_mcp" / "main.py"
SIM_MAIN = HERE.parent / "ai2thor-mcp" / "python" / "ai2thor_mcp" / "main.py"
TOOLS = (
    "get_current_view",
    "set_nav_goal",
    "get_nav_status",
    "cancel_nav",
    "list_destinations",
    "get_map",
    "do_move",
    "do_rotate",
    "do_look",
    "set_attention",
    "cancel_attention",
    "get_attention_status",
)


def _decorator_summary(node) -> str:
    parts = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call) and getattr(dec.func, "attr", None) in ("tool", "resource"):
            keywords = sorted(f"{k.arg}={ast.dump(k.value)}" for k in dec.keywords)
            parts.append(f"{dec.func.attr}({', '.join(keywords)})")
    return "; ".join(parts)


def describe(path: Path) -> Dict[str, Tuple[str, str, str, Optional[str]]]:
    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TOOLS:
            args = ast.dump(node.args)
            returns = ast.dump(node.returns) if node.returns is not None else "None"
            out[node.name] = (args, returns, _decorator_summary(node), ast.get_docstring(node, clean=False))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=Path, default=SIM_MAIN)
    parser.add_argument("--robot", type=Path, default=ROBOT_MAIN)
    args = parser.parse_args()
    sim, robot = describe(args.sim), describe(args.robot)
    failures = 0
    for name in TOOLS:
        if name not in sim:
            print(f"{name}: missing from {args.sim}")
            failures += 1
            continue
        if name not in robot:
            print(f"{name}: missing from {args.robot}")
            failures += 1
            continue
        labels = ("arguments", "return annotation", "decorator", "docstring")
        for label, a, b in zip(labels, sim[name], robot[name]):
            if a != b:
                failures += 1
                print(f"{name}: {label} differs")
                for line in difflib.unified_diff((a or "").splitlines(), (b or "").splitlines(), "sim", "robot", lineterm=""):
                    print("   " + line)
    print(f"{len(TOOLS) - failures}/{len(TOOLS)} tools identical" if not failures else f"{failures} difference(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
