from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import state as st
from .config import AppConfig
from .yolo_format import format_yolo_pose_label, padded_bbox


def _stable_split(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    return "val" if (h % 5 == 0) else "train"


def export_route_object(folder: Path, route: str, obj_type: str, config: AppConfig, out_root: Path) -> int:
    labels_dir = folder / "labels"
    if not labels_dir.is_dir():
        return 0
    # Only training_labels are written to the exported dataset -- e.g. SFP
    # plug's a9-a12 are real measured points used to make the auto-calc fit
    # more robust (non-coplanar reference sets), but are not themselves
    # training targets; a1-a8 are. Completeness ("is this camera done?")
    # still checks against ALL labels via triplet status, since the guided
    # flow only marks an object DONE once every point (a1-a12) is placed.
    label_defs = config.training_labels(route, obj_type)
    dataset_dir = out_root / f"{route}_{obj_type}"
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    count = 0
    manifest_rows = []
    for json_path in sorted(labels_dir.glob("*.json")):
        triplet = st.TripletState.load(json_path)
        for camera in ("left", "center", "right"):
            cs = triplet.cameras[camera]
            obj_state = cs[obj_type]
            if obj_state["status"] != st.STATUS_DONE:
                continue
            img_path = triplet.image_paths.get(camera)
            if not img_path or not Path(img_path).is_file():
                continue
            points = obj_state["points"]
            if not all(l in points for l in label_defs):
                continue
            image = cv2.imread(img_path)
            if image is None:
                continue
            h, w = image.shape[:2]
            kps = np.asarray([[points[l]["x"], points[l]["y"]] for l in label_defs], dtype=np.float64)
            bbox = padded_bbox(kps, w, h)
            if bbox is None:
                continue
            line = format_yolo_pose_label(bbox, kps, w, h, class_id=0)

            split = _stable_split(f"{triplet.stem}_{camera}")
            out_name = f"{triplet.stem}__{camera}"
            dst_img = dataset_dir / "images" / split / f"{out_name}.jpg"
            dst_lbl = dataset_dir / "labels" / split / f"{out_name}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text(line + "\n")
            count += 1
            manifest_rows.append({
                "stem": triplet.stem, "camera": camera, "split": split,
                "sources": {l: points[l]["source"] for l in label_defs},
            })

    yaml_path = dataset_dir / "dataset.yaml"
    names_block = f"  0: {route}_{obj_type}"
    n_kp = len(label_defs)
    yaml_path.write_text(
        f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\n"
        f"names:\n{names_block}\n\nkpt_shape: [{n_kp}, 3]\n"
    )
    with open(dataset_dir / "manifest.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)
    return count
