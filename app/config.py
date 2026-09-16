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

    def local_keypoints(self, route: str, obj_type: str) -> Optional[np.ndarray]:
        lk = self.object_config(route, obj_type)["local_keypoints_m"]
        return np.asarray(lk, dtype=np.float64) if lk is not None else None

    def reference_points(self, route: str, obj_type: str) -> Optional[dict]:
        return self.object_config(route, obj_type)["reference_points"]

    def auto_calculate(self, route: str, obj_type: str) -> bool:
        return bool(self.object_config(route, obj_type)["auto_calculate"])

    def reference_image_path(self, route: str, obj_type: str, camera: str) -> Path:
        rel = self.object_config(route, obj_type)["reference_image_dir"]
        return APP_ROOT / rel / f"{camera}.png"

    def color(self, name: str) -> str:
        return self.data["colors"][name]


def load_calibration(path: Path) -> dict[str, CameraCalibration]:
    d = json.loads(path.read_text())
    return {cam: CameraCalibration.from_dict(cfg) for cam, cfg in d["cameras"].items()}
