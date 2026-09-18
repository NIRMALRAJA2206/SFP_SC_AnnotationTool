"""Resumable per-triplet labeling state.

One JSON file per triplet under <folder>/labels/<stem>.json. Written
atomically (temp file + os.replace) on every point placement so a crash or
force-quit never corrupts progress. Re-opening the same folder scans
labels/ and resumes at the first triplet that is not fully done.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

STATUS_NOT_STARTED = "not_started"
STATUS_PARTIAL = "partial"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"


def _empty_object_state() -> dict:
    return {"status": STATUS_NOT_STARTED, "points": {}}  # label -> {x,y,source}


def _empty_camera_state() -> dict:
    return {
        "status": STATUS_NOT_STARTED,  # overall for this camera image (skipped/done/partial)
        "plug": _empty_object_state(),
        "port": _empty_object_state(),
    }


class TripletState:
    def __init__(self, path: Path, stem: str, route: str, image_paths: Dict[str, str]):
        self.path = path
        self.stem = stem
        self.route = route
        self.image_paths = image_paths
        self.cameras: Dict[str, dict] = {
            cam: _empty_camera_state() for cam in ("left", "center", "right")
        }

    @classmethod
    def new(cls, labels_dir: Path, stem: str, route: str, image_paths: Dict[str, str]) -> "TripletState":
        return cls(labels_dir / f"{stem}.json", stem, route, image_paths)

    @classmethod
    def load(cls, path: Path) -> "TripletState":
        d = json.loads(path.read_text())
        obj = cls(path, d["stem"], d["route"], d["images"])
        obj.cameras = d["cameras"]
        return obj

    def save(self) -> None:
        payload = {
            "stem": self.stem,
            "route": self.route,
            "images": self.image_paths,
            "cameras": self.cameras,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    def is_fully_resolved(self) -> bool:
        for cam in ("left", "center", "right"):
            cs = self.cameras[cam]
            if cs["status"] == STATUS_SKIPPED:
                continue
            for obj in ("plug", "port"):
                st = cs[obj]["status"]
                if st not in (STATUS_DONE, STATUS_SKIPPED):
                    return False
        return True

    def set_point(self, camera: str, obj_type: str, label: str, x: float, y: float, source: str) -> None:
        self.cameras[camera][obj_type]["points"][label] = {"x": x, "y": y, "source": source}
        self.save()

    def clear_point(self, camera: str, obj_type: str, label: str) -> None:
        self.cameras[camera][obj_type]["points"].pop(label, None)
        self.save()

    def mark_object_status(self, camera: str, obj_type: str, status: str) -> None:
        self.cameras[camera][obj_type]["status"] = status
        self.save()

    def mark_camera_status(self, camera: str, status: str) -> None:
        self.cameras[camera]["status"] = status
        self.save()

    def get_points(self, camera: str, obj_type: str) -> Dict[str, dict]:
        return self.cameras[camera][obj_type]["points"]


def discover_triplets(folder: Path) -> List[Dict[str, str]]:
    """Find every stem present in all 3 of left/center/right (missing a view
    is allowed -- that view is just recorded as absent and must be skipped)."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    by_cam: Dict[str, Dict[str, Path]] = {}
    for cam in ("left", "center", "right"):
        cam_dir = folder / cam
        stems = {}
        if cam_dir.is_dir():
            for p in sorted(cam_dir.iterdir()):
                if p.suffix.lower() in exts:
                    stems[p.stem] = p
        by_cam[cam] = stems

    all_stems = sorted(set(by_cam["left"]) | set(by_cam["center"]) | set(by_cam["right"]))
    triplets = []
    for stem in all_stems:
        images = {}
        for cam in ("left", "center", "right"):
            p = by_cam[cam].get(stem)
            images[cam] = str(p) if p else ""
        triplets.append({"stem": stem, "images": images})
    return triplets


def load_or_init_states(folder: Path, route: str) -> List[TripletState]:
    labels_dir = folder / "labels"
    labels_dir.mkdir(exist_ok=True)
    triplets = discover_triplets(folder)
    states = []
    for t in triplets:
        path = labels_dir / f"{t['stem']}.json"
        if path.is_file():
            states.append(TripletState.load(path))
        else:
            st = TripletState.new(labels_dir, t["stem"], route, t["images"])
            for cam, img_path in t["images"].items():
                if not img_path:
                    st.cameras[cam]["status"] = STATUS_SKIPPED
                    st.cameras[cam]["plug"]["status"] = STATUS_SKIPPED
                    st.cameras[cam]["port"]["status"] = STATUS_SKIPPED
            st.save()
            states.append(st)
    return states


def first_unresolved_index(states: List[TripletState]) -> int:
    if not states:
        return -1  # caller must check for this -- no triplets found at all
    for i, s in enumerate(states):
        if not s.is_fully_resolved():
            return i
    return len(states) - 1
