"""gpt-pet-mcp: the real GPTPet robot behind ai2thor-mcp's tool contract.

Every tool below carries the simulator's signature and docstring verbatim (ai2thor-mcp
`python/ai2thor_mcp/main.py`); FastMCP derives the schemas from them, so both servers list the same
tools. Only the bodies differ: they hand the work to `Robot` (robot.py) in a worker thread, so a
drive that takes seconds never blocks the other clients' calls (the portal, the trace sampler).

Tool definitions stay above the `if __name__ == "__main__":` guard (they register on import).
The robot is created lazily on the first call, so this module imports (and lists its tools)
without rclpy; `main()` starts it before serving.
"""

from __future__ import annotations

import argparse
import io
import os
import signal
import sys
import threading
from typing import Any, Callable, Dict, Optional

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from fastmcp.utilities.types import Image
from loguru import logger
from PIL import Image as PILImage

from gpt_pet_mcp import config, contract
from gpt_pet_mcp.errors import RobotError
from gpt_pet_mcp.model import LOOK_DIRECTION, MOVE_ACTION, ROTATE_ACTION

_ROBOT: Any = None
_ROBOT_LOCK = threading.Lock()


def robot() -> Any:
    """The started Robot, created on first use (rclpy is imported here, not at module import)."""
    global _ROBOT
    with _ROBOT_LOCK:
        if _ROBOT is None:
            from gpt_pet_mcp.robot import Robot

            instance = Robot()
            instance.start()
            _ROBOT = instance
        return _ROBOT


async def _run(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a robot method in a worker thread; its RobotErrors become MCP tool errors."""
    try:
        return await anyio.to_thread.run_sync(fn, *args)
    except RobotError as exc:
        raise ToolError(str(exc)) from None


def _image_from(pil_image: PILImage.Image, fmt: str) -> Image:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return Image(data=buf.getvalue(), format=fmt.lower())


mcp = FastMCP(
    name="GptPetMcpServer",
    instructions=(
        "Control GPTPet, a camera-only home robot (ROS 2 Nav2 + SLAM, a Kinect a foot above the floor). "
        "Call get_current_view to see the world and its selectable destinations, "
        "then set_nav_goal with a destination id to drive there; poll get_nav_status "
        "and use cancel_nav to stop. do_move/do_rotate give manual control; the camera "
        "sits about 0.3 m above the floor and do_look tilts it up 27 degrees or straight. "
        "set_attention keeps a person framed in the background (turn, tilt, standoff)."
    ),
)


@mcp.resource(
    uri="view://rgb",
    name="RGB View",
    description="Provides view from robots RGB camera",
    mime_type="image/png",
)
async def rgb_view():
    logger.debug("Fetching RGB view from the camera")
    png = await anyio.to_thread.run_sync(lambda: robot().rgb_png())
    if png is None:
        raise ResourceError("No camera frame: the Kinect is not publishing.")
    return png


@mcp.tool(output_schema=None)
async def get_current_view():
    """Gets the robot's current camera view, annotated with selectable destinations.

    Returns the RGB frame with numbered markers plus a JSON list of destinations:
    visible objects (green boxes) and drivable passages detected from depth
    (cyan boxes: doorways, hallways, wide openings that fit the robot). Pass any
    destination "id" to set_nav_goal to drive there. Marker numbers restart each
    view; ids are stable.
    """
    annotated, payload = await _run(lambda: robot().current_view())
    return [_image_from(annotated, "JPEG"), payload]


@mcp.tool()
async def set_nav_goal(
    destination_id: str, speed_mps: float = config.DEFAULT_SPEED_MPS
) -> dict:
    """Plans a path to a destination and starts driving there in the background.

    destination_id comes from get_current_view / list_destinations (e.g.
    "obj:Fridge|-01.50|+00.00|-00.70" or "psg:1"). Returns immediately with the
    planned path and ETA; poll get_nav_status for progress. An active goal is
    preempted (canceled and replaced), like nav2.

    Args:
        destination_id (str): id of the destination to drive to
        speed_mps (float, optional): driving speed in meters/second. Defaults to 0.35.
    """
    return await _run(lambda: robot().set_nav_goal(destination_id, speed_mps))


@mcp.tool()
async def get_nav_status() -> dict:
    """Gets the state of the current/last navigation goal.

    state is one of idle | planning | following | succeeded | canceled | failed;
    progress holds distance traveled/remaining, percent, eta_s, and waypoint
    counts while a goal exists. Poll this after set_nav_goal until a terminal
    state (succeeded/canceled/failed).
    """
    return await _run(lambda: robot().nav_status())


@mcp.tool()
async def cancel_nav() -> dict:
    """Cancels the active navigation goal, stopping the robot within one step."""
    return await _run(lambda: robot().cancel_nav())


@mcp.tool()
async def list_destinations() -> dict:
    """Lists every destination the robot has seen so far (across all views).

    Ids remain valid for set_nav_goal: objects are re-resolved to their current
    position; passages are fixed world waypoints.
    """
    return await _run(lambda: robot().list_destinations())


@mcp.tool()
async def get_map() -> Image:
    """Gets the top-down occupancy map the robot has built from its depth camera at camera
    height, like a lidar SLAM map: unexplored grey, free space white, obstacles black. Overlaid:
    robot pose (red triangle), remaining planned path (yellow), known destinations (dots), and
    the active goal (star). Unexplored areas fill in as the robot looks around and drives."""
    top_view = await _run(lambda: robot().map_image())
    return _image_from(top_view, "PNG")


@mcp.tool()
async def do_move(action: MOVE_ACTION, moveMagnitude: float = 1.0):
    """Moves the agent ahead("MoveAhead"), back("MoveBack"), left("MoveLeft"), or right("MoveRight"). Returns a summary of result of moving.

    Manual moves preempt (cancel) any active navigation goal, like nav2 teleop.

    Example: do_move(action="MoveAhead", moveMagnitude=2) -> moves agent forward by 2 meters

    Args:
        action (MOVE_ACTION): direction to move
        moveMagnitude (float, optional): how much to move by in meters. Defaults to 1..
    """
    return await _run(lambda: robot().do_move(action, moveMagnitude))


@mcp.tool()
async def do_rotate(action: ROTATE_ACTION, degrees: float = 90.0):
    """Rotates the agent left or right. Returns a summary of the result of rotating.

    Manual rotations preempt (cancel) any active navigation goal, like nav2 teleop.

    Example: do_rotate(action="RotateRight", degrees=45) -> rotates agent 45 degrees to the right

    Args:
        action (ROTATE_ACTION): rotation direction/action (e.g. "RotateLeft" or "RotateRight")
        degrees (float, optional): how many degrees to rotate. Defaults to 90..
    """
    return await _run(lambda: robot().do_rotate(action, degrees))


@mcp.tool()
async def do_look(direction: LOOK_DIRECTION) -> str:
    """Tilts the robot's camera, like the Kinect on the real robot. The camera sits about
    0.3 m above the floor: level, it sees legs, furniture bases and the floor ahead. "up"
    tilts it 27 degrees up to see faces, tabletops and what a person is doing; "straight"
    levels it again. The tilt stays until changed, except that starting a drive levels the
    camera. Returns the camera's new tilt, not a picture: call get_current_view to see.

    Args:
        direction (LOOK_DIRECTION): "up" or "straight"
    """
    return await _run(lambda: robot().do_look(direction))


# --- attention: the robot keeps a person framed on its own (robot contract) ---------------------


@mcp.tool()
async def set_attention(
    destination_id: str, standoff_m: Optional[float] = None, follow: bool = False, greet: bool = False
) -> dict:
    """Keeps a person in view in the background, like the robot's own reflex: it turns to
    centre them (within 10 degrees), tilts the camera up when they are closer than 2 m and
    level beyond 2.5 m, drives to about 1.3 m when they stand still farther than 1.8 m away,
    and backs up when they are closer than 0.9 m. Only a person can be attended: pass the
    Person destination id from get_current_view. Ends by itself after 5 s out of sight
    ("lost"), and whenever a drive (set_nav_goal) or manual control (do_move, do_rotate,
    do_look) starts; an active drive is preempted when attention starts.

    Returns the attention status, also inside every get_current_view as "attention":
    state (off | aiming | approaching | framed | searching | lost), distance_m, bearing_deg
    (positive = to the right), seen_s (in view for), unseen_s, moving, tilt_deg (camera),
    standoff_m, since_s, reason (why it ended). "framed" means the view is centred on the
    person with the right tilt: a good moment to call get_current_view and look.

    Args:
        destination_id (str): the person's destination id, e.g. "obj:person_0"
        standoff_m (float): how close to stand, 0.7 to 2.0 m (default 1.3); it backs up when
            they are 0.4 m closer than that
        follow (bool): keep up with them when they walk off (approach a walking person too)
        greet (bool): once they are framed, wiggle (turn a little left, right and back): hello
    """
    return await _run(lambda: robot().set_attention(destination_id, standoff_m, follow, greet))


@mcp.tool()
async def cancel_attention() -> dict:
    """Stops attending to the person (the camera keeps its tilt; do_look levels it).
    Returns the final attention status."""
    return await _run(lambda: robot().cancel_attention())


@mcp.tool()
async def get_attention_status() -> dict:
    """The attention status without a picture (see set_attention): state, target, distance_m,
    bearing_deg, seen_s, unseen_s, moving, tilt_deg, standoff_m, since_s, reason."""
    return await _run(lambda: robot().attention_status())


if not contract.ATTENTION_ENABLED:
    # the attention tools join the served contract with the person detector (Phase 4)
    for _name in contract.ATTENTION_TOOLS:
        mcp.remove_tool(_name)


def _install_signal_handlers() -> None:
    def _stop(signum, frame) -> None:  # noqa: ARG001
        logger.info("Signal {}: stopping", signum)
        try:
            if _ROBOT is not None:
                _ROBOT.stop()
        finally:
            sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except ValueError:
            pass  # not the main thread


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="MCP server for the real GPTPet robot (ROS 2)")
    parser.add_argument("--http", action="store_true", help="use streamable-http transport")
    parser.add_argument("--host", default="0.0.0.0", help="bind address for --http (default 0.0.0.0)")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("FASTMCP_PORT", "8000")), help="port for --http (FASTMCP_PORT, default 8000)"
    )
    parser.add_argument("--no-robot", action="store_true", help="serve without starting ROS (schemas only; every tool errors)")
    args = parser.parse_args(argv)

    logger.info(
        "gpt-pet-mcp: camera {} m up, {} m forward; frames {} px wide; detector {}; speed cap {} m/s",
        config.CAMERA_HEIGHT_M,
        config.CAMERA_FORWARD_M,
        config.FRAME_WIDTH_PX,
        config.DETECTOR_MODEL,
        config.ROBOT_MAX_SPEED_MPS,
    )
    if not args.no_robot:
        robot()
    _install_signal_handlers()
    try:
        if args.http:
            mcp.run(transport="streamable-http", host=args.host, port=args.port, stateless_http=True)
        else:
            mcp.run()
    finally:
        if _ROBOT is not None:
            _ROBOT.stop()


if __name__ == "__main__":
    main()
