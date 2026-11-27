"""The Kinect: frames for the views and the tilt motor for do_look.

`CameraFeed` subscribes to the RGB and depth topics (compressed by default: JPEG and 16-bit PNG
from image_transport_plugins on the Jetson, a few hundred kB per pair instead of a megabyte and a
half of raw pixels over WiFi) and to the camera_info. Callbacks only keep the raw messages; a tool
thread decodes the latest pair on demand, scales it to FRAME_WIDTH_PX wide (500x375) and scales
the intrinsics with it, so the brain's pixel-to-bearing arithmetic (a 500 px frame) holds.

`KinectTilt` drives the Kinect's motor through the driver node's `tilt_deg` parameter and keeps
the commanded angle, which is what the views report (the brain expects exactly 27.0 or 0.0); the
measured angle from `tilt_state` is only logged.
"""

from __future__ import annotations

import io
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image as PILImage
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float64
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from gpt_pet_mcp import config
from gpt_pet_mcp.perception import Intrinsics

CAMERA_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
"""Best effort: over WiFi a dropped frame beats a retransmitted stale one. Matches a reliable publisher."""
COMPRESSED_DEPTH_HEADER_BYTES = 12  # compressed_depth_image_transport ConfigHeader: int32 format + 2 x float32


class CameraError(RuntimeError):
    """No usable frame: the message names the topics so the operator can check the Jetson."""


class TiltError(RuntimeError):
    """The tilt motor could not be driven: the message names the service."""


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8, FRAME_WIDTH_PX wide
    depth_m: np.ndarray  # (H, W) float32 metres, NaN where the Kinect saw nothing
    intr: Intrinsics  # scaled to the frame
    stamp_s: float  # arrival time (monotonic)

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.stamp_s


@dataclass
class _Stamped:
    msg: Any
    arrived: float  # monotonic
    stamp: float  # header stamp in seconds, 0.0 when the driver left it empty


def _header_s(msg: Any) -> float:
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def decode_rgb(msg: Any) -> np.ndarray:
    """sensor_msgs/CompressedImage (JPEG/PNG) or sensor_msgs/Image (rgb8/bgr8) -> (H, W, 3) uint8 RGB."""
    if isinstance(msg, CompressedImage):
        image = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        return np.asarray(image, dtype=np.uint8)
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    encoding = (msg.encoding or "rgb8").lower()
    if encoding in ("rgb8", "bgr8"):
        array = data.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return array[:, :, ::-1].copy() if encoding == "bgr8" else array.copy()
    if encoding in ("rgba8", "bgra8"):
        array = data.reshape(msg.height, msg.step)[:, : msg.width * 4].reshape(msg.height, msg.width, 4)[:, :, :3]
        return array[:, :, ::-1].copy() if encoding == "bgra8" else array.copy()
    if encoding in ("mono8", "8uc1"):
        array = data.reshape(msg.height, msg.step)[:, : msg.width]
        return np.repeat(array[:, :, None], 3, axis=2)
    raise CameraError(f"unsupported RGB encoding {msg.encoding!r}")


def decode_depth_m(msg: Any) -> np.ndarray:
    """Depth in metres as float32, NaN where invalid. Takes compressedDepth (PNG of the 16-bit
    millimetre image behind the transport's 12-byte header) or a raw mono16/16UC1/32FC1 image."""
    if isinstance(msg, CompressedImage):
        raw = bytes(msg.data)
        fmt = (msg.format or "").lower()
        if "compresseddepth" in fmt:
            if len(raw) <= COMPRESSED_DEPTH_HEADER_BYTES:
                raise CameraError("empty compressedDepth message")
            header = struct.unpack("<iff", raw[:COMPRESSED_DEPTH_HEADER_BYTES])  # format, depthQuantA, depthQuantB
            payload = raw[COMPRESSED_DEPTH_HEADER_BYTES:]
            image = PILImage.open(io.BytesIO(payload))
            array = np.asarray(image)
            if "32fc1" in fmt:
                # inverse-depth quantisation: depth = depthQuantA / (value - depthQuantB)
                quant_a, quant_b = float(header[1]), float(header[2])
                values = array.astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    depth = np.where(values > 0, quant_a / (values - quant_b), np.nan).astype(np.float32)
                return depth
            depth = array.astype(np.float32) / 1000.0
            depth[array == 0] = np.nan
            return depth
        image = PILImage.open(io.BytesIO(raw))
        array = np.asarray(image)
        depth = array.astype(np.float32) / 1000.0
        depth[array == 0] = np.nan
        return depth
    encoding = (msg.encoding or "16UC1").lower()
    data = bytes(msg.data)
    if encoding in ("mono16", "16uc1"):
        dtype = ">u2" if msg.is_bigendian else "<u2"
        array = np.frombuffer(data, dtype=dtype).reshape(msg.height, msg.step // 2)[:, : msg.width]
        depth = array.astype(np.float32) / 1000.0
        depth[array == 0] = np.nan
        return depth
    if encoding == "32fc1":
        dtype = ">f4" if msg.is_bigendian else "<f4"
        array = np.frombuffer(data, dtype=dtype).reshape(msg.height, msg.step // 4)[:, : msg.width]
        depth = array.astype(np.float32).copy()
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        return depth
    raise CameraError(f"unsupported depth encoding {msg.encoding!r}")


def intrinsics_from_camera_info(msg: Optional[CameraInfo]) -> Intrinsics:
    if msg is None or msg.width == 0 or float(msg.k[0]) == 0.0:
        fx, fy, cx, cy = config.CAMERA_INFO_FALLBACK
        return Intrinsics(width=config.KINECT_WIDTH_PX, height=config.KINECT_HEIGHT_PX, fx=fx, fy=fy, cx=cx, cy=cy)
    k = msg.k
    return Intrinsics(width=int(msg.width), height=int(msg.height), fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))


def scale_frame(rgb: np.ndarray, depth: np.ndarray, intr: Intrinsics, width_px: int) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    """Resize the pair to `width_px` wide (RGB bilinear, depth nearest) and scale the intrinsics."""
    src_h, src_w = rgb.shape[:2]
    if src_w == width_px:
        return rgb, depth, intr
    sx = width_px / src_w
    height_px = max(1, int(round(src_h * sx)))
    sy = height_px / src_h
    rgb_small = np.asarray(PILImage.fromarray(rgb).resize((width_px, height_px), PILImage.BILINEAR), dtype=np.uint8)
    rows = np.minimum((np.arange(height_px) / sy).astype(int), depth.shape[0] - 1)
    cols = np.minimum((np.arange(width_px) / sx).astype(int), depth.shape[1] - 1)
    depth_small = depth[rows[:, None], cols[None, :]]
    scaled = Intrinsics(width=width_px, height=height_px, fx=intr.fx * sx, fy=intr.fy * sy, cx=intr.cx * sx, cy=intr.cy * sy)
    return rgb_small, depth_small, scaled


class CameraFeed:
    """Latest RGB + depth pair, decoded on demand. `attach(bridge)` creates the subscriptions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: Deque[_Stamped] = deque(maxlen=6)
        self._depth: Deque[_Stamped] = deque(maxlen=6)
        self._info: Optional[CameraInfo] = None
        self._frames = 0
        self._decoded: Optional[Tuple[Tuple[float, float], Frame]] = None
        self.rgb_topic = config.RGB_TOPIC
        self.depth_topic = config.DEPTH_TOPIC
        if config.CAMERA_TRANSPORT == "compressed":
            self.rgb_topic = f"{config.RGB_TOPIC}/compressed"
            self.depth_topic = f"{config.DEPTH_TOPIC}/compressedDepth"

    def attach(self, bridge: Any) -> None:
        node = bridge.node
        group = bridge.group
        rgb_type = CompressedImage if config.CAMERA_TRANSPORT == "compressed" else Image
        node.create_subscription(rgb_type, self.rgb_topic, self._on_rgb, CAMERA_QOS, callback_group=group)
        node.create_subscription(rgb_type, self.depth_topic, self._on_depth, CAMERA_QOS, callback_group=group)
        node.create_subscription(CameraInfo, config.CAMERA_INFO_TOPIC, self._on_info, CAMERA_QOS, callback_group=group)
        logger.info("Camera: {} + {} ({} transport)", self.rgb_topic, self.depth_topic, config.CAMERA_TRANSPORT)

    def _on_rgb(self, msg: Any) -> None:
        with self._lock:
            self._rgb.append(_Stamped(msg, time.monotonic(), _header_s(msg)))
            self._frames += 1

    def _on_depth(self, msg: Any) -> None:
        with self._lock:
            self._depth.append(_Stamped(msg, time.monotonic(), _header_s(msg)))

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._info = msg

    def _pair(self) -> Tuple[Optional[_Stamped], Optional[_Stamped]]:
        """The latest RGB and the depth closest to it in time: by header stamps when the driver
        sets them, else by arrival (the Kinect emits both at the same rate)."""
        with self._lock:
            if not self._rgb or not self._depth:
                return (self._rgb[-1] if self._rgb else None), (self._depth[-1] if self._depth else None)
            rgb = self._rgb[-1]
            if rgb.stamp > 0 and all(d.stamp > 0 for d in self._depth):
                depth = min(self._depth, key=lambda d: abs(d.stamp - rgb.stamp))
            else:
                depth = min(self._depth, key=lambda d: abs(d.arrived - rgb.arrived))
            return rgb, depth

    def latest_pair(self, max_age_s: float = config.FRAME_MAX_AGE_S) -> Frame:
        """The current frame pair, scaled; raises CameraError when nothing fresh arrived."""
        rgb, depth = self._pair()
        now = time.monotonic()
        if rgb is None or now - rgb.arrived > max_age_s:
            raise CameraError(
                f"No camera frame: the Kinect is not publishing ({self.rgb_topic}, {self.depth_topic}). Drive tools still work."
            )
        if depth is None or now - depth.arrived > max_age_s + config.FRAME_SYNC_SLOP_S * 5:
            raise CameraError(
                f"No depth frame: the Kinect publishes {self.rgb_topic} but not {self.depth_topic}. Drive tools still work."
            )
        key = (rgb.arrived, depth.arrived)
        with self._lock:
            cached = self._decoded
            info = self._info
        if cached is not None and cached[0] == key:
            return cached[1]
        rgb_array = decode_rgb(rgb.msg)
        depth_array = decode_depth_m(depth.msg)
        intr = intrinsics_from_camera_info(info)
        if depth_array.shape[:2] != rgb_array.shape[:2]:
            depth_array = np.asarray(
                PILImage.fromarray(depth_array).resize((rgb_array.shape[1], rgb_array.shape[0]), PILImage.NEAREST),
                dtype=np.float32,
            )
        if intr.width != rgb_array.shape[1] or intr.height != rgb_array.shape[0]:
            # camera_info for another resolution: scale it to the pixels we have
            sx, sy = rgb_array.shape[1] / intr.width, rgb_array.shape[0] / intr.height
            intr = Intrinsics(rgb_array.shape[1], rgb_array.shape[0], intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy)
        rgb_s, depth_s, intr_s = scale_frame(rgb_array, depth_array, intr, config.FRAME_WIDTH_PX)
        frame = Frame(rgb=rgb_s, depth_m=depth_s, intr=intr_s, stamp_s=rgb.arrived)
        with self._lock:
            self._decoded = (key, frame)
        return frame

    def latest_rgb_png(self) -> Optional[bytes]:
        try:
            frame = self.latest_pair(max_age_s=60.0)
        except CameraError:
            return None
        buf = io.BytesIO()
        PILImage.fromarray(frame.rgb).save(buf, format="PNG")
        return buf.getvalue()

    def health(self) -> Dict[str, Any]:
        with self._lock:
            rgb = self._rgb[-1] if self._rgb else None
            depth = self._depth[-1] if self._depth else None
            info = self._info is not None
            frames = self._frames
        now = time.monotonic()
        return {
            "rgb_age_s": None if rgb is None else round(now - rgb.arrived, 2),
            "depth_age_s": None if depth is None else round(now - depth.arrived, 2),
            "intrinsics": info,
            "frames": frames,
            "topics": [self.rgb_topic, self.depth_topic, config.CAMERA_INFO_TOPIC],
        }


class KinectTilt:
    """The Kinect's tilt motor through the driver node's `tilt_deg` parameter."""

    def __init__(self, bridge: Any, node_name: str = config.KINECT_NODE) -> None:
        self._bridge = bridge
        self._node_name = node_name
        self._lock = threading.Lock()
        self._commanded = 0.0
        self._measured: Optional[float] = None
        self._measured_at: Optional[float] = None

    def attach(self, bridge: Any) -> None:
        bridge.node.create_subscription(Float64, config.TILT_STATE_TOPIC, self._on_state, 10, callback_group=bridge.group)

    def _on_state(self, msg: Float64) -> None:
        with self._lock:
            self._measured = float(msg.data)
            self._measured_at = time.monotonic()

    def tilt_deg(self) -> float:
        with self._lock:
            return self._commanded

    def measured(self) -> Optional[float]:
        with self._lock:
            if self._measured_at is None or time.monotonic() - self._measured_at > 5.0:
                return None
            return self._measured

    def ready(self) -> bool:
        return self._bridge.parameter_service_ready(self._node_name)

    def set(self, degrees: float) -> None:
        ok, reason = self._bridge.set_parameter(self._node_name, "tilt_deg", float(degrees), config.TILT_TIMEOUT_S)
        if not ok:
            if "unavailable" in reason:
                raise TiltError(f"the Kinect node is not running ({self._node_name}/set_parameters unavailable)")
            raise TiltError(reason)
        with self._lock:
            self._commanded = float(degrees)
        logger.info("Kinect tilt -> {:g} deg (measured {})", degrees, self.measured())
