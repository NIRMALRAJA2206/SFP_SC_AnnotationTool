"""Standalone re-implementation of the canonical YOLO-pose label format used
across the project (rrsl_sfp_plug_geometry.py / mtu_sc_plug_geometry.py /
mtu_sc_mouth_geometry.py::format_yolo_pose_label / padded_bbox /
visibility_flags), so this tool has no dependency on the aic_serl package."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def padded_bbox(pixels: np.ndarray, image_width: int, image_height: int,
                 padding: float = 0.25, min_side_px: float = 16.0):
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    w = float(x_max - x_min)
    h = float(y_max - y_min)
    if w <= 0.0 or h <= 0.0:
        return None
    x_min = max(0.0, float(x_min - padding * w))
    x_max = min(float(image_width - 1), float(x_max + padding * w))
    y_min = max(0.0, float(y_min - padding * h))
    y_max = min(float(image_height - 1), float(y_max + padding * h))
    if (x_max - x_min) < min_side_px or (y_max - y_min) < min_side_px:
        return None
    return x_min, y_min, x_max, y_max


def format_yolo_pose_label(bbox_xyxy: Sequence[float], pixels: np.ndarray,
                            image_width: int, image_height: int, *, class_id: int = 0,
                            visibility: int = 2) -> str:
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2).copy()
    width, height = float(image_width), float(image_height)
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)

    x_min, y_min, x_max, y_max = [float(v) for v in bbox_xyxy]
    tokens = [
        str(int(class_id)),
        f"{((x_min + x_max) * 0.5) / width:.7f}",
        f"{((y_min + y_max) * 0.5) / height:.7f}",
        f"{(x_max - x_min) / width:.7f}",
        f"{(y_max - y_min) / height:.7f}",
    ]
    for px, py in points:
        tokens.extend([f"{px / width:.7f}", f"{py / height:.7f}", str(int(visibility))])
    return " ".join(tokens)
