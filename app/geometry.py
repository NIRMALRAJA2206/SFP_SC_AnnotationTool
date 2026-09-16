"""Reference-point -> full-keypoint-set math.

Ported and generalized from the validated SFP/SC plug math built earlier
(run_sfp_reference_point_math.py / run_sc_reference_point_math.py). Handles
any per-camera reference-point count/geometry, not just the original 3-point
coplanar case:

  - Exactly 3 reference points -> P3P. If those 3 are coplanar (the original
    default config), there is a genuine 2-fold mirror-branch ambiguity
    (solveP3P returns up to 4 solutions; reprojection error on the 3 points
    used to solve is ~0 for every branch -- it cannot distinguish the real
    one on its own).
  - >=4 reference points that are still coplanar -> IPPE, which explicitly
    returns its own 2-fold ambiguity as 2 branches (same underlying issue,
    just solved with more points).
  - >=4 reference points that are NOT coplanar (e.g. a config that mixes a
    near-face point with a depth-extended point, so the set spans more than
    one plane) -> SQPNP, a single well-posed solution with no mirror
    ambiguity to resolve.
  - Whatever the branch count per camera, disambiguation always works the
    same way: >= 2 cameras that each have all their reference points placed
    contribute their branch(es); every (branch, branch) pair across those
    cameras is converted into a common frame (tool0) and the pair with the
    SMALLEST disagreement is kept. A correct pair agrees to a few mm / ~1
    degree; a mirror-mismatched pair disagrees by tens of mm / tens of
    degrees. A camera with only 1 or 2 reference points can never
    independently resolve a pose; it only *receives* the fused pose,
    transformed into its own camera frame via the calibrated rig extrinsics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    K: np.ndarray
    dist: np.ndarray
    T_tool0_from_optical: np.ndarray  # 4x4

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibration":
        return cls(
            K=np.asarray(d["K"], dtype=np.float64),
            dist=np.asarray(d.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64),
            T_tool0_from_optical=np.asarray(d["T_tool0_from_optical"], dtype=np.float64),
        )


def compute_direct_midpoints(
    midpoint_of: Dict[str, List[str]],
    points_by_camera_px: Dict[str, Dict[str, Tuple[float, float]]],
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Fill any midpoint-defined label directly as the 2-D pixel average of
    its two parent labels, PER CAMERA, wherever both parents are already
    placed in that same image.

    This deliberately needs no calibration and does no 3-D triangulation --
    it is a genuinely different (and, for small close-range objects, often
    more trustworthy) computation than reprojecting a fitted 3-D pose: it
    can't inherit calibration error, cross-camera disagreement, or a wrong
    assumed rigid-body dimension, because it never leaves 2-D pixel space
    for that camera. The tradeoff, real and worth stating: a 2-D pixel
    midpoint is not exactly the projection of the true 3-D midpoint under
    perspective (only exact for orthographic projection or a fronto-parallel
    line) -- for a small object at a working-distance standoff the
    perspective error is normally far smaller than the calibration/geometry
    errors it avoids, but it is not zero.

    Returns only the newly-computed {camera: {label: (x, y)}} entries --
    never overwrites an already-placed point (manual or otherwise); callers
    decide what to do with a label that has no direct midpoint available in
    a given camera (fall back to the calibrated multi-view pipeline).
    """
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for cam, existing in points_by_camera_px.items():
        cam_out = {}
        for target, (parent_a, parent_b) in midpoint_of.items():
            if target in existing:
                continue
            if parent_a in existing and parent_b in existing:
                ax, ay = existing[parent_a]
                bx, by = existing[parent_b]
                cam_out[target] = ((ax + bx) / 2.0, (ay + by) / 2.0)
        if cam_out:
            out[cam] = cam_out
    return out


@dataclass
class AutoCalcResult:
    success: bool
    reason: str = ""
    points_by_camera: Dict[str, np.ndarray] = field(default_factory=dict)  # cam -> (N,2) px
    position_disagreement_mm: Optional[float] = None
    rotation_disagreement_deg: Optional[float] = None
    used_cameras: Tuple[str, str] = ()
    warning: bool = False


def _camera_matrix(K: np.ndarray, T_tool0_from_cam: np.ndarray) -> np.ndarray:
    T_cam_from_tool0 = np.linalg.inv(T_tool0_from_cam)
    return K @ T_cam_from_tool0[:3, :4]


def _is_coplanar(pts: np.ndarray, tol: float = 1e-4) -> bool:
    """True if pts (Nx3) lie (near-)exactly on a common plane -- checked via
    the smallest singular value of the centered point set relative to the
    object's own scale, not an absolute threshold."""
    centered = pts - pts.mean(axis=0)
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1e-9)
    s = np.linalg.svd(centered, compute_uv=False)
    return bool(s[-1] / scale < tol)


def _solve_pose_branches(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """Return every mathematically valid (R, t) branch for this point set.

    3 points -> P3P (up to 4 branches, coplanar-mirror ambiguity expected).
    >=4 coplanar points -> IPPE (2 branches -- still ambiguous in general,
    e.g. this project's "center" camera reference set once extended with
    non-coplanar-breaking points can still end up coplanar on its own).
    >=4 non-coplanar points -> SQPNP (1 branch; a well-conditioned
    non-coplanar set has no mirror ambiguity to resolve)."""
    n = len(obj_pts)
    if n == 3:
        _, rvecs, tvecs = cv2.solveP3P(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_AP3P)
    elif _is_coplanar(obj_pts):
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    else:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    branches = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        branches.append((R, t))
    return branches


def _cam_pose_to_tool0(T_tool0_from_cam: np.ndarray, R_cam_obj: np.ndarray, t_cam_obj: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_cam_obj
    T[:3, 3] = t_cam_obj
    return T_tool0_from_cam @ T


def _pose_disagreement(Ta: np.ndarray, Tb: np.ndarray) -> Tuple[float, float]:
    pos_a, pos_b = Ta[:3, 3], Tb[:3, 3]
    Ra, Rb = Ta[:3, :3], Tb[:3, :3]
    R_diff = Ra.T @ Rb
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(pos_a - pos_b) * 1e3), ang


def _rotmat_to_quat(Rm: np.ndarray) -> np.ndarray:
    tr = np.trace(Rm)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (Rm[2, 1] - Rm[1, 2]) / s
        y = (Rm[0, 2] - Rm[2, 0]) / s
        z = (Rm[1, 0] - Rm[0, 1]) / s
    else:
        i = int(np.argmax([Rm[0, 0], Rm[1, 1], Rm[2, 2]]))
        if i == 0:
            s = math.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
            w = (Rm[2, 1] - Rm[1, 2]) / s; x = 0.25 * s
            y = (Rm[0, 1] + Rm[1, 0]) / s; z = (Rm[0, 2] + Rm[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
            w = (Rm[0, 2] - Rm[2, 0]) / s; x = (Rm[0, 1] + Rm[1, 0]) / s
            y = 0.25 * s; z = (Rm[1, 2] + Rm[2, 1]) / s
        else:
            s = math.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
            w = (Rm[1, 0] - Rm[0, 1]) / s; x = (Rm[0, 2] + Rm[2, 0]) / s
            y = (Rm[1, 2] + Rm[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


WARN_MM = 5.0
WARN_DEG = 5.0


def auto_calculate(
    local_keypoints_m: np.ndarray,
    labels: List[str],
    reference_points_cfg: Dict[str, List[str]],
    points_by_camera_px: Dict[str, Dict[str, Tuple[float, float]]],
    calibration: Dict[str, CameraCalibration],
) -> AutoCalcResult:
    """
    local_keypoints_m: (N,3) full local keypoint set.
    labels: length-N label list, same order as local_keypoints_m.
    reference_points_cfg: {camera: [label, ...]} -- which labels are the
        reference (green) points for that camera.
    points_by_camera_px: {camera: {label: (x, y)}} -- ALREADY PLACED points
        only (reference points the user has clicked so far).
    calibration: {camera: CameraCalibration}.

    Returns an AutoCalcResult with reprojected pixel positions for every
    label in every camera that has a calibration entry -- callers overwrite
    only the non-reference ("calculated") slots with these.
    """
    solvable_cams = [
        cam for cam, refs in reference_points_cfg.items()
        if len(refs) >= 3
        and cam in points_by_camera_px
        and all(lbl in points_by_camera_px[cam] for lbl in refs)
        and cam in calibration
    ]
    if len(solvable_cams) < 2:
        return AutoCalcResult(success=False, reason="need >=2 cameras with all reference points placed")

    label_index = {lbl: i for i, lbl in enumerate(labels)}

    branches_by_cam = {}
    for cam in solvable_cams:
        refs = reference_points_cfg[cam]
        obj_pts = np.asarray([local_keypoints_m[label_index[l]] for l in refs], dtype=np.float64)
        img_pts = np.asarray([points_by_camera_px[cam][l] for l in refs], dtype=np.float64)
        calib = calibration[cam]
        try:
            branches_by_cam[cam] = _solve_pose_branches(obj_pts, img_pts, calib.K, calib.dist)
        except cv2.error as exc:
            return AutoCalcResult(success=False, reason=f"pose solve failed on {cam}: {exc}")
        if not branches_by_cam[cam]:
            return AutoCalcResult(success=False, reason=f"pose solve returned no solution on {cam}")

    # Try every pair of solvable cameras, every branch combination; keep the
    # globally best-agreeing (cam_a, branch_i, cam_b, branch_j).
    best = None
    cams = solvable_cams
    for ai in range(len(cams)):
        for bi in range(ai + 1, len(cams)):
            cam_a, cam_b = cams[ai], cams[bi]
            for i, (Ra, ta) in enumerate(branches_by_cam[cam_a]):
                Ta = _cam_pose_to_tool0(calibration[cam_a].T_tool0_from_optical, Ra, ta)
                for j, (Rb, tb) in enumerate(branches_by_cam[cam_b]):
                    Tb = _cam_pose_to_tool0(calibration[cam_b].T_tool0_from_optical, Rb, tb)
                    d_mm, d_deg = _pose_disagreement(Ta, Tb)
                    if best is None or d_mm < best[0]:
                        best = (d_mm, d_deg, cam_a, cam_b, Ta, Tb)

    d_mm, d_deg, cam_a, cam_b, Ta, Tb = best

    q_a, q_b = _rotmat_to_quat(Ta[:3, :3]), _rotmat_to_quat(Tb[:3, :3])
    if np.dot(q_a, q_b) < 0:
        q_b = -q_b
    q_fused = q_a + q_b
    q_fused /= np.linalg.norm(q_fused)
    R_fused = _quat_to_rotmat(q_fused)
    t_fused = (Ta[:3, 3] + Tb[:3, 3]) / 2.0
    T_tool0_obj = np.eye(4)
    T_tool0_obj[:3, :3] = R_fused
    T_tool0_obj[:3, 3] = t_fused

    points_by_camera = {}
    for cam, calib in calibration.items():
        T_cam_obj = np.linalg.inv(calib.T_tool0_from_optical) @ T_tool0_obj
        rvec, _ = cv2.Rodrigues(T_cam_obj[:3, :3])
        tvec = T_cam_obj[:3, 3].reshape(3, 1)
        proj, _ = cv2.projectPoints(local_keypoints_m, rvec, tvec, calib.K, calib.dist)
        points_by_camera[cam] = proj.reshape(-1, 2)

    return AutoCalcResult(
        success=True,
        points_by_camera=points_by_camera,
        position_disagreement_mm=d_mm,
        rotation_disagreement_deg=d_deg,
        used_cameras=(cam_a, cam_b),
        warning=(d_mm > WARN_MM or d_deg > WARN_DEG),
    )
