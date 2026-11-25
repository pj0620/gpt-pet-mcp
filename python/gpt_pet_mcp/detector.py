"""Object detection on pj-ubuntu's GPU: an open-vocabulary YOLO-World model (Ultralytics) with a
household vocabulary whose prompts map to AI2-THOR object types, so the destinations the brain
sees ("Sofa", "Fridge", "Person") read the same from both servers.

The model loads lazily in the background at startup; until it is loaded (or when the `detector`
extra is not installed, or GPTPET_MCP_DETECTOR=none) views carry passages only, and the health log
says why.
"""

from __future__ import annotations

import json
import threading
import time
from importlib import resources
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from gpt_pet_mcp import config
from gpt_pet_mcp.objects import Detection


def load_vocabulary(path: Optional[str] = None) -> Dict[str, str]:
    """prompt -> AI2-THOR label. The packaged vocabulary.json unless a path is given."""
    if path:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(resources.files("gpt_pet_mcp").joinpath("vocabulary.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError("the vocabulary must map prompt strings to label strings")
    return data


class ObjectDetector:
    def __init__(
        self,
        model_name: str = config.DETECTOR_MODEL,
        vocabulary: Optional[Dict[str, str]] = None,
        conf: float = config.DETECTOR_CONF,
        device: str = config.DETECTOR_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.device = device
        self._vocabulary = vocabulary
        self._prompts: List[str] = []
        self._model: Any = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._loading = False
        self._loaded_at: Optional[float] = None
        self._last_ms: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return self.model_name.strip().lower() not in ("", "none", "off")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load_in_background(self) -> None:
        if not self.enabled or self._loading or self.loaded:
            return
        self._loading = True
        threading.Thread(target=self.load, name="detector-load", daemon=True).start()

    def load(self) -> bool:
        """Import Ultralytics, load the weights, set the vocabulary, warm up. False with the reason kept."""
        if not self.enabled:
            return False
        try:
            from ultralytics import YOLO  # the `detector` extra

            vocabulary = self._vocabulary or load_vocabulary(config.VOCABULARY_PATH)
            prompts = list(vocabulary.keys())
            started = time.monotonic()
            model = YOLO(self.model_name)
            if hasattr(model, "set_classes"):
                model.set_classes(prompts)
            model.predict(np.zeros((375, 500, 3), dtype=np.uint8), conf=self.conf, device=self.device, verbose=False, imgsz=640)
            with self._lock:
                self._vocabulary = vocabulary
                self._prompts = prompts
                self._model = model
                self._error = None
                self._loaded_at = time.monotonic()
            logger.info("Detector {} on {} ready in {:.1f} s ({} prompts)", self.model_name, self.device, time.monotonic() - started, len(prompts))
            return True
        except Exception as exc:  # noqa: BLE001 - a missing GPU or extra must not stop the server
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("Detector unavailable ({}); views carry passages only", self._error)
            return False
        finally:
            self._loading = False

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        """Boxes in the frame's pixels with AI2-THOR labels; [] while the detector is not ready."""
        with self._lock:
            model = self._model
            vocabulary = self._vocabulary or {}
            prompts = self._prompts
        if model is None:
            return []
        started = time.monotonic()
        # Ultralytics treats numpy arrays as BGR (OpenCV's order)
        results = model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), conf=self.conf, device=self.device, verbose=False, imgsz=640)
        detections: List[Detection] = []
        height, width = rgb.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", None) or {}
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                index = int(cls)
                prompt = names.get(index) if isinstance(names, dict) else None
                if prompt is None and 0 <= index < len(prompts):
                    prompt = prompts[index]
                label = vocabulary.get(str(prompt), None) if prompt is not None else None
                if label is None:
                    label = str(prompt).title().replace(" ", "") if prompt is not None else "Object"
                bbox = (
                    int(max(0, min(width - 1, round(float(x1))))),
                    int(max(0, min(height - 1, round(float(y1))))),
                    int(max(0, min(width - 1, round(float(x2))))),
                    int(max(0, min(height - 1, round(float(y2))))),
                )
                detections.append(Detection(label=label, conf=float(conf), bbox=bbox))
        self._last_ms = (time.monotonic() - started) * 1000.0
        return detections

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "loaded": self._model is not None,
                "loading": self._loading,
                "model": self.model_name,
                "device": self.device,
                "prompts": len(self._prompts),
                "last_ms": None if self._last_ms is None else round(self._last_ms, 1),
                "error": self._error,
            }
