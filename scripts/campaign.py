#!/usr/bin/env python3
"""The robot tool test campaign driver: one subcommand per station, every tool call logged with
the Mac's wall clock (the iPad clip clock), every view and map saved, events polled from the tools
so nobody has to say "now". Writes ~/robot-rig/campaign/<date>/<station>.jsonl and prints the
ipad_clip.sh line that cuts the station's clip.

    uv run --extra dev python scripts/campaign.py s1 --round 1
    uv run --extra dev python scripts/campaign.py s8 --chair obj:chair#1 --box obj:box#1

Env: GPTPET_MCP_ROBOT_URL (default http://pj-ubuntu.local:8000/mcp), CAMPAIGN_DIR, PJSSH (an ssh
command for pj-ubuntu, used for pose_truth.py and depth_rows.py; default `ssh pj-ubuntu.local`).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.exceptions import ToolError

URL = os.environ.get("GPTPET_MCP_ROBOT_URL", "http://pj-ubuntu.local:8000/mcp")
OUT = Path(os.environ.get("CAMPAIGN_DIR", f"~/robot-rig/campaign/{dt.date.today().isoformat()}")).expanduser()
PJSSH = os.environ.get("PJSSH", "ssh pj-ubuntu.local")
REMOTE_PREFIX = "export PATH=$HOME/.local/bin:$PATH; source /opt/ros/humble/setup.bash; source ~/gpt-pet-mcp/env/pj-ubuntu.env; cd ~/gpt-pet-mcp && uv run python"
TERMINAL = {"succeeded", "failed", "canceled"}


def clock(t: Optional[float] = None) -> str:
    return dt.datetime.fromtimestamp(t if t is not None else time.time()).strftime("%H:%M:%S.%f")[:-3]


def parse_text(result) -> Any:
    texts = [b.text for b in result.content if getattr(b, "type", None) == "text"]
    if not texts:
        return None
    try:
        return json.loads(texts[0])
    except ValueError:
        return texts[0]


def unwrap(value: Any) -> Any:
    return value["result"] if isinstance(value, dict) and set(value) == {"result"} else value


class Station:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = OUT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / f"{name}.jsonl"
        self.t_start = time.time()
        self.images = 0
        self.client: Optional[Client] = None
        self.log("start", url=URL)
        print(f"== {name} started {clock(self.t_start)} -> {self.dir}")

    async def __aenter__(self) -> "Station":
        self.client = Client(URL, timeout=240)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self.client is not None:
            await self.client.__aexit__(*exc)
        self.finish()

    def log(self, kind: str, **data: Any) -> None:
        record = {"t": time.time(), "clock": clock(), "kind": kind, **data}
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def cue(self, text: str) -> None:
        self.log("cue", text=text)
        print(f"\n>>> CUE {clock()}: {text}\n", flush=True)

    def note(self, text: str, **data: Any) -> None:
        self.log("note", text=text, **data)
        print(f"   {clock()} {text}", flush=True)

    async def call(self, tool: str, args: Optional[dict] = None, tag: str = "") -> Any:
        """Call a tool; return its payload (dict, str or {'error': text}); save image blocks."""
        args = args or {}
        t0 = time.time()
        try:
            result = await self.client.call_tool(tool, args, raise_on_error=True)
        except ToolError as exc:
            self.log("error", tool=tool, args=args, text=str(exc), ms=round((time.time() - t0) * 1000))
            print(f"   {clock()} {tool}({args}) -> ERROR {exc}", flush=True)
            return {"error": str(exc)}
        payload = unwrap(parse_text(result))
        images = []
        for block in result.content:
            if getattr(block, "type", None) == "image":
                self.images += 1
                ext = "png" if block.mimeType == "image/png" else "jpg"
                path = self.dir / f"{self.images:03d}_{tool}{('_' + tag) if tag else ''}.{ext}"
                path.write_bytes(base64.b64decode(block.data))
                images.append(path.name)
        self.log("call", tool=tool, args=args, ms=round((time.time() - t0) * 1000), payload=payload, images=images)
        shown = payload if isinstance(payload, str) else json.dumps({k: v for k, v in payload.items() if k not in ("destinations",)} if isinstance(payload, dict) else payload)[:160]
        print(f"   {clock()} {tool}({json.dumps(args)}) -> {shown}", flush=True)
        return payload

    async def view(self, tag: str = "") -> dict:
        payload = await self.call("get_current_view", {}, tag)
        if isinstance(payload, dict) and "destinations" in payload:
            for d in payload["destinations"]:
                print(f"        {d['marker']:2d} {d['id']:16s} {d.get('distance_m')} m  {d['description']}  bbox={d.get('bbox') or d.get('region')}")
        return payload

    async def status(self) -> dict:
        return await self.call("get_nav_status")

    async def attention(self, quiet: bool = False) -> dict:
        t0 = time.time()
        result = await self.client.call_tool("get_attention_status", {}, raise_on_error=True)
        payload = unwrap(parse_text(result))
        self.log("attn", ms=round((time.time() - t0) * 1000), payload=payload)
        if not quiet:
            print(f"   {clock()} attention {payload.get('state')} d={payload.get('distance_m')} b={payload.get('bearing_deg')} tilt={payload.get('tilt_deg')}")
        return payload

    async def rgb_resource(self, tag: str = "rgb") -> Optional[str]:
        contents = await self.client.read_resource("view://rgb")
        for item in contents:
            blob = getattr(item, "blob", None)
            if blob:
                path = self.dir / f"resource_{tag}.png"
                path.write_bytes(base64.b64decode(blob))
                self.log("resource", uri="view://rgb", bytes=path.stat().st_size, file=path.name)
                return path.name
        return None

    def pose_truth(self, label: str) -> Optional[dict]:
        cmd = f"{PJSSH} {shlex.quote(REMOTE_PREFIX + ' scripts/pose_truth.py --label ' + shlex.quote(label))}"
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
            data = json.loads(out[-1]) if out else None
        except Exception as exc:  # noqa: BLE001
            data = {"error": str(exc)}
        self.log("pose_truth", label=label, data=data)
        if data and data.get("slam"):
            print(f"   {clock()} pose[{label}] slam={data['contract']} wheel_yaw={round(data['wheel']['yaw_deg'], 1) if data.get('wheel') else None} ekf_yaw={round(data['ekf']['yaw_deg'], 1) if data.get('ekf') else None}")
        return data

    def depth_rows(self, save: Optional[str] = None) -> Optional[dict]:
        extra = f" --save {shlex.quote(save)}" if save else ""
        cmd = f"{PJSSH} {shlex.quote(REMOTE_PREFIX + ' scripts/depth_rows.py' + extra)}"
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90).stdout.strip().splitlines()
            data = json.loads(out[-1]) if out else None
        except Exception as exc:  # noqa: BLE001
            data = {"error": str(exc)}
        self.log("depth_rows", save=save, data=data)
        if data:
            print(f"   {clock()} depth rows top={data.get('top_rows')} bottom={data.get('bottom_rows')} floor={data.get('floor')} fwd={data.get('camera_forward_est_m')}")
        return data

    async def wait_label(self, label: str, timeout_s: float, every_s: float = 1.5, min_d: float = 0.0, max_d: float = 99.0) -> Optional[dict]:
        """Poll views until a destination with `label` (within a distance band) appears."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            payload = await self.view(f"wait_{label.lower()}")
            if isinstance(payload, dict):
                hits = [d for d in payload.get("destinations", []) if d["label"].lower() == label.lower() and min_d <= d.get("distance_m", 0.0) <= max_d]
                if hits:
                    return min(hits, key=lambda d: d.get("distance_m", 9.0))
            await asyncio.sleep(every_s)
        self.note(f"no {label} seen within {timeout_s:g} s")
        return None

    async def wait_nav(self, timeout_s: float = 150.0, every_s: float = 0.5, views_every_s: float = 0.0) -> List[dict]:
        """Poll get_nav_status to a terminal state; optional views in between (keep tracks fresh)."""
        seen: List[dict] = []
        deadline = time.monotonic() + timeout_s
        last_view = time.monotonic()
        while time.monotonic() < deadline:
            st = await self.status()
            seen.append(st)
            if st.get("state") in TERMINAL or st.get("state") == "idle":
                return seen
            if views_every_s and time.monotonic() - last_view >= views_every_s:
                await self.view("during_drive")
                last_view = time.monotonic()
            await asyncio.sleep(every_s)
        self.note(f"nav did not end within {timeout_s:g} s; cancelling")
        await self.call("cancel_nav")
        return seen

    async def attention_watch(self, stop: asyncio.Event, every_s: float = 0.25, view_every_s: float = 3.0) -> List[dict]:
        samples: List[dict] = []
        last_view = time.monotonic()
        while not stop.is_set():
            try:
                samples.append(await self.attention(quiet=len(samples) % 8 != 0))
            except Exception as exc:  # noqa: BLE001
                self.log("error", tool="get_attention_status", text=str(exc))
            if view_every_s and time.monotonic() - last_view >= view_every_s:
                try:
                    await self.view("attn")
                except Exception as exc:  # noqa: BLE001
                    self.log("error", tool="get_current_view", text=str(exc))
                last_view = time.monotonic()
            await asyncio.sleep(every_s)
        return samples

    async def wait_attention(self, predicate, timeout_s: float, every_s: float = 0.25) -> Optional[dict]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = await self.attention(quiet=True)
            if predicate(st):
                return st
            await asyncio.sleep(every_s)
        return None

    def finish(self) -> None:
        t_end = time.time()
        self.log("end", seconds=round(t_end - self.t_start, 1))
        start = clock(self.t_start - 2.0)[:8]
        secs = int(t_end - self.t_start + 4)
        print(f"== {self.name} done {clock(t_end)} ({secs - 4} s)")
        print(f"   clip: bin/video/ipad_clip.sh {start} {secs} ~/robot-rig/campaign/{OUT.name}/{self.name}.mp4")


def yaw_delta(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


# --- stations -------------------------------------------------------------------------------------


async def s1(args) -> None:
    rounds = {1: ["Box", "Backpack", "Chair"], 2: ["Bottle", "Ball", "HousePlant", "Shoe"]}
    expected = rounds[args.round]
    marks = {1: {"Box": 1.0, "Backpack": 2.0, "Chair": 3.0}, 2: {"Bottle": 1.0, "Ball": 2.0, "HousePlant": 3.0, "Shoe": math.hypot(2.0, 1.0)}}[args.round]
    async with Station(f"s1_r{args.round}") as st:
        st.cue(f"round {args.round}: " + ", ".join(f"{k} on {v if k != 'Shoe' else 'the left tick'}" for k, v in marks.items()) + "; then step out of the picture")
        st.note("waiting for the first of the expected labels")
        await st.wait_label(expected[0], 180.0, every_s=2.0)
        views = []
        for i in range(3):
            views.append(await st.view(f"v{i + 1}"))
            await asyncio.sleep(2.0)
        st.depth_rows(save=f"regression/data/robot_frames/s1_r{args.round}")
        known = await st.call("list_destinations")
        await st.rgb_resource(f"s1_r{args.round}")
        hits: Dict[str, List[float]] = {label: [] for label in expected}
        extras: Dict[str, int] = {}
        bearings: Dict[str, List[float]] = {}
        for v in views:
            for d in v.get("destinations", []):
                if d["label"] in hits:
                    hits[d["label"]].append(d.get("distance_m"))
                    if d.get("bbox"):
                        u = (d["bbox"][0] + d["bbox"][2]) / 2.0
                        bearings.setdefault(d["label"], []).append(round(math.degrees(math.atan((u - 250.0) / (525.0 * 500 / 640))), 1))
                elif d["kind"] == "object":
                    extras[d["label"]] = extras.get(d["label"], 0) + 1
        summary = {label: {"hits": len(hits[label]), "distances": hits[label], "mark_m": marks[label], "bearings_deg": bearings.get(label)} for label in expected}
        summary["false_labels_per_view"] = round(sum(extras.values()) / 3.0, 2)
        summary["extras"] = extras
        ids = [sorted(d["id"] for d in v.get("destinations", [])) for v in views]
        summary["ids_stable"] = all(i == ids[0] for i in ids)
        summary["known"] = len(known.get("destinations", [])) if isinstance(known, dict) else None
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2))


async def s2(args) -> None:
    async with Station(f"s2_{args.mark:g}m") as st:
        st.cue(f"stand on the {args.mark:g} mark facing the robot, still, until the camera has tilted up and down twice")
        person = await st.wait_label("Person", 180.0, every_s=2.0)
        out = []
        for cycle in range(2):
            for direction in ("straight", "up", "straight"):
                text = await st.call("do_look", {"direction": direction})
                v = await st.view(f"c{cycle}_{direction}")
                rows = st.depth_rows(save=f"regression/data/robot_frames/s2_{args.mark:g}m_{direction}_{cycle}" if cycle == 0 else None)
                people = [d for d in v.get("destinations", []) if d["label"] == "Person"]
                out.append({
                    "direction": direction, "text": text, "camera": v.get("camera"),
                    "person_bbox": people[0]["bbox"] if people else None, "person_distance": people[0].get("distance_m") if people else None,
                    "bottom_rows": rows.get("bottom_rows") if rows else None, "tilt_measured": rows.get("tilt_measured_deg") if rows else None,
                })
                await asyncio.sleep(1.0)
        rot = await st.call("do_rotate", {"action": "RotateLeft", "degrees": 10})
        after = await st.view("after_rotate")
        st.log("summary", steps=out, rotate_keeps_tilt=after.get("camera"), rotate_text=rot, first_person=person)
        print(json.dumps({"steps": out, "camera_after_rotate": after.get("camera")}, indent=2))


async def s3(args) -> None:
    async with Station("s3") as st:
        st.cue("doorway empty, nobody in the picture")
        far = [await st.view(f"far{i}") for i in range(3)]
        st.depth_rows(save="regression/data/robot_frames/s3_far")
        st.pose_truth("s3 before move")
        await st.call("do_move", {"action": "MoveAhead", "moveMagnitude": 1.0})
        st.pose_truth("s3 after move")
        await asyncio.sleep(1.0)
        near = [await st.view(f"near{i}") for i in range(3)]
        st.depth_rows(save="regression/data/robot_frames/s3_near")
        side = "RotateLeft" if args.wall == "left" else "RotateRight"
        back = "RotateRight" if args.wall == "left" else "RotateLeft"
        await st.call("do_rotate", {"action": side, "degrees": 30})
        wall = await st.view("wall")
        st.depth_rows(save="regression/data/robot_frames/s3_wall")
        await st.call("do_rotate", {"action": back, "degrees": 30})
        known = await st.call("list_destinations")

        def passages(v):
            return [(d["id"], d["label"], d["description"], d.get("distance_m")) for d in v.get("destinations", []) if d["kind"] == "passage"]

        summary = {"far": [passages(v) for v in far], "near": [passages(v) for v in near], "wall": passages(wall), "known_passages": [d for d in known.get("destinations", []) if d["kind"] == "passage"]}
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2))


async def s4(args) -> None:
    async with Station("s4") as st:
        await st.call("get_map", {}, "baseline")
        known = await st.call("list_destinations")
        st.log("summary", known=known)
        print(json.dumps(known, indent=2)[:2000])


async def s5(args) -> None:
    seq = [("RotateLeft", 90), ("RotateRight", 90), ("RotateLeft", 30), ("RotateRight", 30)] + [("RotateLeft", 90)] * 4
    async with Station("s5") as st:
        st.cue("1 m clear around the robot; it spins eight times")
        await st.call("get_map", {}, "before")
        rows = []
        for action, deg in seq:
            before = st.pose_truth(f"before {action} {deg}")
            a0 = (await st.status())["agent"]
            text = await st.call("do_rotate", {"action": action, "degrees": deg})
            await asyncio.sleep(1.0)
            a1 = (await st.status())["agent"]
            after = st.pose_truth(f"after {action} {deg}")
            expected = deg if action == "RotateRight" else -deg
            row = {"action": action, "deg": deg, "text": text, "slam_delta": round(yaw_delta(a0["yaw"], a1["yaw"]), 1) if a0 and a1 else None, "expected": expected}
            for src in ("wheel", "ekf", "slam"):
                if before and after and before.get(src) and after.get(src):
                    row[f"{src}_delta_ccw"] = round(yaw_delta(before[src]["yaw_deg"], after[src]["yaw_deg"]), 1)
            rows.append(row)
            print(f"   -> {row}")
        await st.call("get_map", {}, "after")
        ratios = [abs(r["slam_delta"]) / r["deg"] for r in rows if r.get("slam_delta") is not None]
        mean = sum(ratios) / len(ratios) if ratios else None
        cv = (math.sqrt(sum((x - mean) ** 2 for x in ratios) / len(ratios)) / mean) if ratios and mean else None
        summary = {"rows": rows, "ratio_mean": round(mean, 3) if mean else None, "ratio_cv": round(cv, 3) if cv else None}
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2))


async def s6(args) -> None:
    seq = [("MoveAhead", 0.5), ("MoveBack", 0.5), ("MoveRight", 0.3), ("MoveLeft", 0.3), ("MoveAhead", 1.0), ("MoveBack", 1.0)]
    async with Station("s6") as st:
        st.cue("1.5 m clear ahead and 0.7 m each side; six short moves")
        rows = []
        for action, mag in seq:
            before = st.pose_truth(f"before {action} {mag}")
            text = await st.call("do_move", {"action": action, "moveMagnitude": mag})
            await asyncio.sleep(1.0)
            after = st.pose_truth(f"after {action} {mag}")
            row = {"action": action, "m": mag, "text": text}
            for src in ("wheel", "ekf", "slam"):
                if before and after and before.get(src) and after.get(src):
                    dx, dy = after[src]["x"] - before[src]["x"], after[src]["y"] - before[src]["y"]
                    th = math.radians(before[src]["yaw_deg"])
                    row[f"{src}_fwd"] = round(dx * math.cos(th) + dy * math.sin(th), 3)
                    row[f"{src}_left"] = round(-dx * math.sin(th) + dy * math.cos(th), 3)
                    row[f"{src}_dyaw"] = round(yaw_delta(before[src]["yaw_deg"], after[src]["yaw_deg"]), 1)
            rows.append(row)
            print(f"   -> {row}")
        st.log("summary", rows=rows)
        print(json.dumps(rows, indent=2))


async def s7(args) -> None:
    async with Station("s7") as st:
        st.cue("put the box with its front on the 0.5 mark, then step back")
        box = await st.wait_label("Box", 120.0, every_s=2.0, max_d=1.2)
        before = st.pose_truth("before blocked move")
        text = await st.call("do_move", {"action": "MoveAhead", "moveMagnitude": 2.0})
        after = st.pose_truth("after blocked move")
        moved = None
        if before and after and before.get("slam") and after.get("slam"):
            moved = round(math.hypot(after["slam"]["x"] - before["slam"]["x"], after["slam"]["y"] - before["slam"]["y"]), 3)
        match = re.match(r"^Moved (\d+\.\d\d) of 2\.00 m ahead, then blocked by (.+)$", text or "")
        st.note(f"blocked-move text ok: {bool(match)}; slam moved {moved} m")
        back = await st.call("do_move", {"action": "MoveBack", "moveMagnitude": 0.25})
        st.cue("take the box away")
        st.log("summary", box=box, text=text, moved_slam=moved, text_ok=bool(match), back=back)


async def s8(args) -> None:
    async with Station("s8") as st:
        st.cue("chair on the 2.5 mark, box 1 m to the right of the 1.5 mark; then out of the picture")
        chair = await st.wait_label("Chair", 180.0, every_s=2.0)
        box = next((d for d in (await st.view("find_box")).get("destinations", []) if d["label"] == "Box"), None)
        if not chair:
            return
        err = await st.call("set_nav_goal", {"destination_id": "obj:nope"})
        await st.call("do_look", {"direction": "up"})
        a = await st.call("set_nav_goal", {"destination_id": chair["id"]})
        v = await st.view("after_goal_a")
        await asyncio.sleep(2.0)
        b = await st.call("set_nav_goal", {"destination_id": box["id"]}) if box else {"error": "no box seen"}
        await asyncio.sleep(1.0)
        mid = await st.status()
        await asyncio.sleep(2.0)
        cancel = await st.call("cancel_nav")
        after_cancel = await st.status()
        await asyncio.sleep(2.0)
        st.note("full drive to the chair")
        full = await st.call("set_nav_goal", {"destination_id": chair["id"]})
        seen = await st.wait_nav(150.0)
        final = await st.view("arrived")
        st.pose_truth("after full drive")
        summary = {
            "unknown_error": err, "tilt_after_goal": v.get("camera"), "a": a, "b": b, "b_preempts_a": isinstance(b, dict) and b.get("preempted_goal_id") == (a.get("goal_id") if isinstance(a, dict) else None),
            "status_during_b": mid.get("goal"), "cancel": cancel, "state_after_cancel": after_cancel.get("state"),
            "full": full, "full_states": [s["state"] for s in seen], "full_traveled": [s["progress"]["distance_traveled_m"] for s in seen if s.get("progress")],
            "final_reason": seen[-1].get("reason") if seen else None, "chair_in_final_view": [d for d in final.get("destinations", []) if d["label"] == "Chair"],
        }
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2, default=str)[:3000])
        if args.speed:
            st.note("speed check: 0.05 m/s for 10 s then cancel (robot must be back near HOME first)")
            slow = await st.call("set_nav_goal", {"destination_id": chair["id"], "speed_mps": 0.05})
            t0 = time.time()
            await asyncio.sleep(10.0)
            s_slow = await st.status()
            await st.call("cancel_nav")
            st.log("summary_speed", slow=slow, traveled_10s=s_slow.get("progress"), elapsed=round(time.time() - t0, 1))
            print("slow drive 10 s:", s_slow.get("progress"))


async def s9(args) -> None:
    async with Station(f"s9{args.step}") as st:
        goal = args.goal
        if not goal:
            v = await st.view("find_passage")
            psg = [d for d in v.get("destinations", []) if d["kind"] == "passage"]
            goal = psg[0]["id"] if psg else None
        if not goal:
            st.note("no passage in view; pass --goal <id>")
            return
        cues = {
            "a": "the moment the robot rolls, stand in the doorway and stay there",
            "b": "block the doorway again; when it stops in front of you count 8 s, then step aside",
            "c": "when it starts rolling, put the box about 1 m ahead of it; take it away once it has stopped or passed",
        }
        st.cue(cues[args.step])
        started = await st.call("set_nav_goal", {"destination_id": goal})
        seen = await st.wait_nav(150.0, views_every_s=2.0)
        st.pose_truth(f"s9{args.step} end")
        summary = {"goal": goal, "started": started, "states": [s["state"] for s in seen][-20:], "final": seen[-1] if seen else None, "seconds": round(time.time() - st.t_start, 1)}
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2, default=str)[:2500])


async def s10(args) -> None:
    async with Station("s10") as st:
        st.cue("stand on the 3.0 mark facing the robot, still")
        person = await st.wait_label("Person", 180.0, every_s=2.0, min_d=2.4, max_d=3.8)
        if not person:
            return
        await st.call("do_rotate", {"action": "RotateLeft", "degrees": 50})
        await asyncio.sleep(1.0)
        stop = asyncio.Event()
        watch = asyncio.create_task(st.attention_watch(stop))
        t0 = time.time()
        first = await st.call("set_attention", {"destination_id": person["id"]})
        framed = await st.wait_attention(lambda s: s.get("state") == "framed" and s.get("bearing_deg") is not None and abs(s["bearing_deg"]) <= 10, 15.0)
        t_framed = time.time() - t0 if framed else None
        approaching = await st.wait_attention(lambda s: s.get("state") == "approaching", 20.0)
        t_appr = time.time() - t0 if approaching else None
        done = await st.wait_attention(lambda s: s.get("state") == "framed" and s.get("distance_m") is not None and 0.9 <= s["distance_m"] <= 1.9 and s.get("tilt_deg") == 27.0, 70.0)
        t_done = time.time() - t0 if done else None
        stop.set()
        samples = await watch
        final = await st.view("final")
        summary = {"first": first, "t_framed_s": t_framed, "t_approaching_s": t_appr, "t_done_s": t_done, "done": done, "person_in_final": [d for d in final.get("destinations", []) if d["label"] == "Person"], "samples": len(samples)}
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2, default=str))


async def s11(args) -> None:
    async with Station(f"s11{args.step}") as st:
        stop = asyncio.Event()
        watch = asyncio.create_task(st.attention_watch(stop))
        if args.step == "a":
            st.cue("step in to about half a metre from the bumper")
            before = (await st.status())["agent"]
            close = await st.wait_attention(lambda s: s.get("distance_m") is not None and s["distance_m"] < 0.85, 30.0)
            recovered = await st.wait_attention(lambda s: s.get("distance_m") is not None and s["distance_m"] >= 0.9 and s.get("state") == "framed", 15.0) if close else None
            after = (await st.status())["agent"]
            moved = math.hypot(after["x"] - before["x"], after["z"] - before["z"]) if before and after else None
            summary = {"close": close, "recovered": recovered, "slam_moved_m": round(moved, 3) if moved is not None else None}
        elif args.step == "b":
            st.cue("walk off sideways briskly, out of its sight, and stay away until the camera levels")
            unseen = await st.wait_attention(lambda s: s.get("unseen_s", 0) > 0.5, 40.0)
            t0 = time.time()
            searching = await st.wait_attention(lambda s: s.get("state") in ("searching", "lost"), 15.0)
            lost = await st.wait_attention(lambda s: s.get("state") == "lost", 40.0)
            v = await st.view("after_lost")
            summary = {"unseen_at": unseen, "searching": searching, "lost": lost, "t_lost_s": round(time.time() - t0, 1) if lost else None, "camera_after": v.get("camera")}
        else:
            st.cue("back on the 3.0 mark; the moment the robot turns toward you, walk straight up to the 1.0 mark at a brisk pace and stop")
            person = await st.wait_label("Person", 120.0, every_s=2.0, min_d=2.3)
            await st.call("do_rotate", {"action": "RotateRight", "degrees": 20})
            first = await st.call("set_attention", {"destination_id": person["id"]}) if person else None
            done = await st.wait_attention(lambda s: s.get("state") == "framed" and s.get("distance_m") is not None and s["distance_m"] <= 1.9 and not s.get("moving"), 45.0)
            summary = {"first": first, "done": done}
        stop.set()
        samples = await watch
        seen = [s for s in samples if s.get("bearing_deg") is not None and s.get("unseen_s", 0) == 0]
        summary["samples"] = len(samples)
        summary["centred_share"] = round(sum(1 for s in seen if abs(s["bearing_deg"]) <= 15) / len(seen), 2) if seen else None
        summary["tilt_up_when_close"] = all(s.get("tilt_deg") == 27.0 for s in seen[1:] if s.get("distance_m") is not None and s["distance_m"] < 1.9) if seen else None
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2, default=str))


async def s12(args) -> None:
    async with Station("s12") as st:
        st.cue("stand on the 2.0 mark; wave both arms from the moment the robot turns to you until it twitches")
        person = await st.wait_label("Person", 180.0, every_s=2.0)
        if not person:
            return
        stop = asyncio.Event()
        watch = asyncio.create_task(st.attention_watch(stop))
        greet = await st.call("set_attention", {"destination_id": person["id"], "greet": True})
        greeted = await st.wait_attention(lambda s: s.get("greeted") is True, 40.0)
        st.note(f"greeted: {bool(greeted)}")
        standoff = await st.call("set_attention", {"destination_id": person["id"], "standoff_m": 0.8})
        near = await st.wait_attention(lambda s: s.get("state") == "framed" and s.get("distance_m") is not None and s["distance_m"] <= 1.1, 60.0)
        bad = await st.call("set_attention", {"destination_id": person["id"], "standoff_m": 0.5})
        st.cue("walk slowly 2 m sideways, then 1 m back, and stop")
        follow = await st.call("set_attention", {"destination_id": person["id"], "follow": True})
        replanned = await st.wait_attention(lambda s: s.get("state") == "approaching", 45.0)
        settled = await st.wait_attention(lambda s: s.get("state") == "framed" and s.get("distance_m") is not None and s["distance_m"] <= 1.9 and not s.get("moving"), 60.0)
        objects = [d for d in (await st.view("before_drive")).get("destinations", []) if d["kind"] == "object" and d["label"] != "Person"]
        drive = await st.call("set_nav_goal", {"destination_id": objects[0]["id"]}) if objects else {"error": "no object to drive to"}
        after_drive = await st.attention()
        await st.call("cancel_nav")
        again = await st.call("set_attention", {"destination_id": person["id"]})
        rot = await st.call("do_rotate", {"action": "RotateLeft", "degrees": 15})
        after_rot = await st.attention()
        await st.call("set_attention", {"destination_id": person["id"]})
        canceled = await st.call("cancel_attention")
        stop.set()
        await watch
        summary = {"greet": greet, "greeted": bool(greeted), "standoff_0_8": near, "standoff_0_5_error": bad, "follow": follow, "replanned": bool(replanned), "settled": settled,
                   "drive": drive, "after_drive": {k: after_drive.get(k) for k in ("state", "reason")}, "rotate_text": rot, "after_rotate": {k: after_rot.get(k) for k in ("state", "reason")}, "cancel": {k: canceled.get(k) for k in ("state", "reason")}}
        st.log("summary", **summary)
        print(json.dumps(summary, indent=2, default=str)[:3000])


async def home(args) -> None:
    """Drive back to HOME (stored by `home --store`) with nav_goal.py on pj-ubuntu."""
    path = OUT / "home.json"
    async with Station("home") as st:
        if args.store:
            agent = (await st.status())["agent"]
            path.write_text(json.dumps(agent))
            st.note(f"HOME stored: {agent}")
            return
        agent = json.loads(path.read_text())
        theta = 90.0 - agent["yaw"]
        cmd = f"{PJSSH} {shlex.quote('source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0; python3 ~/gpt-pet-v2/bin/nav/nav_goal.py ' + f'{agent[chr(120)]} {agent[chr(122)]} {theta} 150')}"
        st.note(f"nav_goal.py to HOME {agent}")
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=200)
        st.log("home", stdout=out.stdout[-2000:], stderr=out.stderr[-500:])
        print(out.stdout[-800:])
        print((await st.status())["agent"])


async def snap(args) -> None:
    async with Station(args.name or "snap") as st:
        await st.view("snap")
        await st.call("get_map", {}, "snap")
        print(json.dumps(await st.status(), indent=1)[:600])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="station", required=True)
    p = sub.add_parser("s1"); p.add_argument("--round", type=int, default=1); p.set_defaults(fn=s1)
    p = sub.add_parser("s2"); p.add_argument("--mark", type=float, default=2.0); p.set_defaults(fn=s2)
    p = sub.add_parser("s3"); p.add_argument("--wall", choices=("left", "right"), default="left"); p.set_defaults(fn=s3)
    sub.add_parser("s4").set_defaults(fn=s4)
    sub.add_parser("s5").set_defaults(fn=s5)
    sub.add_parser("s6").set_defaults(fn=s6)
    sub.add_parser("s7").set_defaults(fn=s7)
    p = sub.add_parser("s8"); p.add_argument("--speed", action="store_true"); p.set_defaults(fn=s8)
    p = sub.add_parser("s9"); p.add_argument("--step", choices=("a", "b", "c"), required=True); p.add_argument("--goal", default=None); p.set_defaults(fn=s9)
    sub.add_parser("s10").set_defaults(fn=s10)
    p = sub.add_parser("s11"); p.add_argument("--step", choices=("a", "b", "c"), required=True); p.set_defaults(fn=s11)
    sub.add_parser("s12").set_defaults(fn=s12)
    p = sub.add_parser("home"); p.add_argument("--store", action="store_true"); p.set_defaults(fn=home)
    p = sub.add_parser("snap"); p.add_argument("--name", default=None); p.set_defaults(fn=snap)
    args = parser.parse_args()
    asyncio.run(args.fn(args))


if __name__ == "__main__":
    main()
