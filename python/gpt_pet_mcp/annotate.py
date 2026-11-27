"""Set-of-marks annotation: draw numbered destination markers on the RGB frame."""

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gpt_pet_mcp import config
from gpt_pet_mcp.describe import DestinationRecord


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=config.FONT_SIZE)
    except TypeError:  # older Pillow: no size kwarg
        return ImageFont.load_default()


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    number: int,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    frame_size: Tuple[int, int],
) -> None:
    r = config.MARKER_RADIUS_PX
    cx = int(np.clip(center[0], r, frame_size[0] - r))
    cy = int(np.clip(center[1], r, frame_size[1] - r))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(255, 255, 255))
    draw.text((cx, cy), str(number), fill=(255, 255, 255), font=font, anchor="mm")


def attention_caption(attention: Dict[str, Any]) -> str:
    """One line for the frame's banner: what attention is doing and to whom."""
    state = attention.get("state") or "off"
    label = attention.get("label") or "person"
    parts = [f"ATTENTION {state}: {label}"]
    distance = attention.get("distance_m")
    if isinstance(distance, (int, float)):
        parts.append(f"{distance:.1f} m")
    bearing = attention.get("bearing_deg")
    if isinstance(bearing, (int, float)):
        parts.append("centred" if abs(bearing) < 1.0 else f"{abs(bearing):.0f} deg {'right' if bearing > 0 else 'left'}")
    if state == "searching":
        parts.append(f"unseen {attention.get('unseen_s', 0):.0f} s")
    elif state == "lost":
        parts.append(str(attention.get("reason") or ""))
    elif attention.get("seen_s"):
        parts.append(f"{attention.get('seen_s', 0):.0f} s")
    tilt = attention.get("tilt_deg")
    if isinstance(tilt, (int, float)) and tilt:
        parts.append(f"camera up {tilt:.0f}")
    return ", ".join(part for part in parts if part)


def _draw_attention(
    draw: ImageDraw.ImageDraw,
    size: Tuple[int, int],
    records: Iterable[DestinationRecord],
    attention: Dict[str, Any],
    font: ImageFont.ImageFont,
) -> None:
    """The attended person's box in the attention colour, and a banner saying what is going on."""
    target = attention.get("target")
    for rec in records:
        if rec.id == target and rec.bbox is not None:
            draw.rectangle(rec.bbox, outline=config.ATTENTION_COLOR, width=4)
    framed = attention.get("state") == "framed"
    height = config.FONT_SIZE + 10
    draw.rectangle((0, 0, size[0], height), fill=config.ATTENTION_COLOR if framed else (40, 40, 40))
    draw.text(
        (8, height // 2),
        attention_caption(attention),
        fill=(255, 255, 255) if framed else config.ATTENTION_COLOR,
        font=font,
        anchor="lm",
    )


def annotate_view(
    frame: np.ndarray, records: Iterable[DestinationRecord], attention: Optional[Dict[str, Any]] = None
) -> Image.Image:
    img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font()
    size = img.size

    for rec in records:
        if rec.marker is None:
            continue
        if rec.kind == "object" and rec.bbox is not None:
            x1, y1, x2, y2 = rec.bbox
            draw.rectangle((x1, y1, x2, y2), outline=config.OBJECT_COLOR, width=2)
            _draw_marker(draw, (x1, y1), rec.marker, config.OBJECT_COLOR, font, size)
        elif rec.kind == "passage":
            if rec.region_px is not None:
                draw.rectangle(rec.region_px, outline=config.PASSAGE_COLOR, width=2)
            anchor: Optional[Tuple[int, int]] = rec.marker_px
            if anchor is None and rec.region_px is not None:
                anchor = (
                    (rec.region_px[0] + rec.region_px[2]) // 2,
                    (rec.region_px[1] + rec.region_px[3]) // 2,
                )
            if anchor is not None:
                _draw_marker(draw, anchor, rec.marker, config.PASSAGE_COLOR, font, size)
    if attention and attention.get("state") not in (None, "off"):
        _draw_attention(draw, size, list(records), attention, font)
    return img
