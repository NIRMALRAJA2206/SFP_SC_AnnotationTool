from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import CameraCalibration

APP_ROOT = Path(__file__).resolve().parent.parent


class AppConfig:
    def __init__(self, path: Path = APP_ROOT / "config.json"):
        self.data = json.loads(path.read_text())

    def object_config(self, route: str, obj_type: str) -> dict:
        return self.data["routes"][route][obj_type]

    def labels(self, route: str, obj_type: str):
        return self.object_config(route, obj_type)["labels"]

    def training_labels(self, route: str, obj_type: str):
        """The subset of `labels` actually written to the exported training
        dataset. Defaults to all labels; a config entry can restrict this
        (e.g. SFP plug uses a1-a12 internally as reference/auto-calc aids
        for a more robust pose fit, but only a1-a8 are real training
        targets -- a9-a12 are extra measured points, not the model's
        output keypoints)."""
        cfg = self.object_config(route, obj_type)
        return cfg.get("training_labels") or cfg["labels"]

    def local_keypoints(self, route: str, obj_type: str) -> Optional[np.ndarray]:
        lk = self.object_config(route, obj_type)["local_keypoints_m"]
        return np.asarray(lk, dtype=np.float64) if lk is not None else None

    def reference_points(self, route: str, obj_type: str) -> Optional[dict]:
        return self.object_config(route, obj_type)["reference_points"]

    def auto_calculate(self, route: str, obj_type: str) -> bool:
        return bool(self.object_config(route, obj_type)["auto_calculate"])

    def midpoint_of(self, route: str, obj_type: str) -> dict:
        """{target_label: [parent_a, parent_b]} -- labels defined as the
        real 3D midpoint of two other labels. Where both parents are
        already placed in the SAME camera view, the target can be filled
        directly as the 2-D pixel midpoint in that image -- no calibration
        or 3-D triangulation needed, and no cross-camera-agreement error to
        inherit. Only labels that can't be filled this way in a given
        camera (a parent missing/not visible there) fall back to the
        calibrated multi-view pipeline."""
        return self.object_config(route, obj_type).get("midpoint_of") or {}

    def rectangles(self, route: str, obj_type: str) -> list:
        """List of 4-label lists, each in cyclic order around one rigid
        rectangle, used for calibration-free parallelogram completion (the
        4th corner from the other 3) -- see geometry.py::
        compute_parallelogram_completion."""
        return self.object_config(route, obj_type).get("rectangles") or []

    def reference_image_path(self, route: str, obj_type: str, camera: str) -> Path:
        rel = self.object_config(route, obj_type)["reference_image_dir"]
        return APP_ROOT / rel / f"{camera}.png"

    def color(self, name: str) -> str:
        return self.data["colors"][name]


def load_calibration(path: Path) -> dict[str, CameraCalibration]:
    d = json.loads(path.read_text())
    return {cam: CameraCalibration.from_dict(cfg) for cam, cfg in d["cameras"].items()}
