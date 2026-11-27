"""Top-down map rendering: slam_toolbox's occupancy grid in the simulator's palette, with the
simulator's overlays (ai2thor-mcp `utils.MapOverlays` / `_draw_map_overlays` and the palette of
`mapping.py`, copied), so `get_map` looks the same from both servers.

Image geometry (the simulator's): PX_PER_M pixels per metre (5 px per 0.05 m cell), image x
along +x and image y downward along +z of the contract frame, which is +ros.y. The grid's row 0
lies at the map origin and rows grow with +y, so the array is drawn as is, row 0 at the top.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from math import cos, radians, sin
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gpt_pet_mcp import config

PX_PER_M = 100
RESOLUTION_M = 0.05
UNKNOWN_GREY = 205
FREE_WHITE = 254
OCCUPIED_BLACK = 0
OCCUPIED_FROM = 50  # occupancy probability (0-100) from which a cell is drawn as an obstacle
MARGIN_M = 0.5  # drawn around the known cells
MAX_RENDER_PX = 2000  # longest side of the PNG; a larger map is downscaled after the overlays


@dataclass
class MapOverlays:
    agent_position: Optional[dict] = None  # {x, y, z} of the robot, contract frame
    agent_yaw_deg: Optional[float] = None
    path: Optional[List[dict]] = None  # remaining waypoint corners, world {x,y,z}
    destinations: Optional[list] = None  # DestinationRecord-like: .target/.kind/.marker
    goal_target: Optional[dict] = None
    attention_target: Optional[dict] = None  # {x, z} of the person the robot is attending to


def m2px(dist: float) -> float:
    return PX_PER_M * dist


def _draw_map_overlays(
    draw: ImageDraw.ImageDraw, overlays: MapOverlays, bound_min_x: float, bound_min_z: float
) -> None:
    def to_px(x: float, z: float) -> tuple[float, float]:
        return m2px(x - bound_min_x), m2px(z - bound_min_z)

    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    if overlays.path and len(overlays.path) >= 1:
        points = [to_px(c["x"], c["z"]) for c in overlays.path]
        if overlays.agent_position is not None:
            points.insert(0, to_px(overlays.agent_position["x"], overlays.agent_position["z"]))
        if len(points) >= 2:
            draw.line(points, fill=config.PATH_COLOR, width=3)

    if overlays.destinations:
        for rec in overlays.destinations:
            target = getattr(rec, "target", None)
            if not target:
                continue
            px, pz = to_px(target["x"], target["z"])
            color = (
                config.PASSAGE_COLOR
                if getattr(rec, "kind", "") == "passage"
                else config.OBJECT_COLOR
            )
            r = 8
            draw.ellipse((px - r, pz - r, px + r, pz + r), fill=color, outline=(255, 255, 255))
            marker = getattr(rec, "marker", None)
            if marker is not None:
                draw.text((px, pz), str(marker), fill=(0, 0, 0), font=font, anchor="mm")

    if overlays.goal_target is not None:
        gx, gz = to_px(overlays.goal_target["x"], overlays.goal_target["z"])
        star = []
        for i in range(10):
            r = 13 if i % 2 == 0 else 5
            angle = radians(i * 36 - 90)
            star.append((gx + r * cos(angle), gz + r * sin(angle)))
        draw.polygon(star, fill=config.GOAL_COLOR, outline=(255, 255, 255))

    if overlays.attention_target is not None:
        tx, tz = to_px(overlays.attention_target["x"], overlays.attention_target["z"])
        if overlays.agent_position is not None:
            ax, az = to_px(overlays.agent_position["x"], overlays.agent_position["z"])
            draw.line([(ax, az), (tx, tz)], fill=config.ATTENTION_COLOR, width=2)
        r = 14
        draw.ellipse((tx - r, tz - r, tx + r, tz + r), outline=config.ATTENTION_COLOR, width=3)

    if overlays.agent_position is not None:
        ax, az = overlays.agent_position["x"], overlays.agent_position["z"]
        yaw = radians(overlays.agent_yaw_deg or 0.0)
        tri = []
        for off_deg, r_m in ((0.0, 0.22), (140.0, 0.14), (-140.0, 0.14)):
            theta = yaw + radians(off_deg)
            tri.append(to_px(ax + r_m * sin(theta), az + r_m * cos(theta)))
        draw.polygon(tri, fill=config.AGENT_COLOR, outline=(255, 255, 255))


def classify(data: np.ndarray) -> np.ndarray:
    """nav_msgs/OccupancyGrid values (-1 unknown, 0-100 occupancy) -> the ROS map palette."""
    image = np.full(data.shape, UNKNOWN_GREY, dtype=np.uint8)
    known = data >= 0
    image[known & (data < OCCUPIED_FROM)] = FREE_WHITE
    image[known & (data >= OCCUPIED_FROM)] = OCCUPIED_BLACK
    return image


class OccupancyGridRenderer:
    """Keeps the latest occupancy grid and renders it on demand (the portal asks every second)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image: Optional[np.ndarray] = None  # classified, (rows, cols)
        self._origin: Tuple[float, float] = (0.0, 0.0)  # world (x, y) of the grid's corner
        self._resolution: float = RESOLUTION_M
        self._updates = 0

    def update(self, width: int, height: int, resolution: float, origin_x: float, origin_y: float, data) -> None:
        grid = np.asarray(data, dtype=np.int8).reshape(int(height), int(width))
        image = classify(grid)
        with self._lock:
            self._image = image
            self._origin = (float(origin_x), float(origin_y))
            self._resolution = float(resolution)
            self._updates += 1

    @property
    def has_map(self) -> bool:
        with self._lock:
            return self._image is not None

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    def known_cells(self) -> int:
        with self._lock:
            return 0 if self._image is None else int(np.count_nonzero(self._image != UNKNOWN_GREY))

    def free_cells_xz(self, data_free_max: int = config.COSTMAP_FREE_MAX) -> Optional[np.ndarray]:
        """Centres (x, z) of the cells drawn free; None before the first grid."""
        with self._lock:
            if self._image is None:
                return None
            rows, cols = np.nonzero(self._image == FREE_WHITE)
            ox, oz = self._origin
            res = self._resolution
        xs = ox + (cols + 0.5) * res
        zs = oz + (rows + 0.5) * res
        return np.stack([xs, zs], axis=1)

    def render(self, overlays: Optional[MapOverlays] = None) -> Image.Image:
        """The map as an RGB image at PX_PER_M, cropped to the known area, with the overlays."""
        with self._lock:
            image = self._image
            ox, oz = self._origin
            res = self._resolution
        if image is None:
            return Image.new("RGB", (100, 100), (UNKNOWN_GREY,) * 3)
        known = image != UNKNOWN_GREY
        margin = int(round(MARGIN_M / res))
        if known.any():
            rows = np.nonzero(known.any(axis=1))[0]
            cols = np.nonzero(known.any(axis=0))[0]
            r0, r1 = max(0, rows[0] - margin), min(image.shape[0], rows[-1] + 1 + margin)
            c0, c1 = max(0, cols[0] - margin), min(image.shape[1], cols[-1] + 1 + margin)
        else:
            r0, r1, c0, c1 = 0, image.shape[0], 0, image.shape[1]
        crop = image[r0:r1, c0:c1]
        scale = max(1, int(round(PX_PER_M * res)))
        img = Image.fromarray(crop, mode="L").convert("RGB")
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        if overlays is not None:
            _draw_map_overlays(ImageDraw.Draw(img), overlays, ox + c0 * res, oz + r0 * res)
        longest = max(img.size)
        if longest > MAX_RENDER_PX:
            factor = MAX_RENDER_PX / longest
            img = img.resize((max(1, int(img.width * factor)), max(1, int(img.height * factor))), Image.NEAREST)
        return img
