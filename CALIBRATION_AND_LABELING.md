# Calibration & Labeling — how this tool actually works

This is the detailed companion to `README.md`. It explains exactly what the
calibration file contains, how those numbers are used at every step of
labeling, and the full worked example (real bugs found and fixed) that
shaped the current design. If you only need install/run steps, use
`README.md`; if you need to understand *why* a point ends up where it does,
or debug a bad result, this is the doc.

---

## 1. The labeling data model

Every route (`sfp`, `sc`) has two objects (`plug`, `port`). Each object has:

- `labels` — ordered list of keypoint names (e.g. SFP plug: `a1..a12`).
- `local_keypoints_m` — the object's **real, physical, rigid-body geometry**:
  each label's `(x, y, z)` position in metres, in the object's own local
  frame. `null` for Port (no 3-D model is used for ports — every point is
  placed manually, always).
- `reference_points` — per camera, which labels are the "green" points a
  human places directly; everything else is either computed or, for Port,
  also placed manually.
- `midpoint_of` (Plug only, optional) — labels defined as the exact 3-D
  midpoint of two other labels (see §4).
- `training_labels` (optional) — the subset of `labels` actually written to
  the exported training dataset; defaults to all of `labels`.
- `auto_calculate` — whether the calibrated pose pipeline (§5) is enabled
  for this object at all.
- `reference_image_dir` — the thumbnail shown while labeling.

**Concrete example — SFP Plug** (`config.json`):
```
a1-a4:  near/tip face corners, 0 in from the tip (the reference depth)
a5-a8:  midpoints of a1-a9, a2-a10, a3-a11, a4-a12 (~0.75in from tip)
a9-a12: a visible reference line/marking, 1.5in from the tip
        (NOT the physical end of the connector -- just a visible landmark)
```
Cross-section: thickness (a1↔a4) = 0.275in = 6.985mm, width (a3↔a4) = 0.5in
= 12.7mm. Both are **real caliper measurements of the physical part**, not
sim-derived guesses — see §7 for how these were verified.

Per-camera reference sets:
```
left:   a2, a3, a4, a10, a11, a12   (6 points)
center: a3, a4, a11, a12            (4 points)
right:  a1, a3, a4, a9, a11, a12    (6 points)
```
These aren't arbitrary — they're whichever corners/marks are actually
visible from each camera angle on the real part.

---

## 2. What "calibration" means here, concretely

A calibration file (`calibration.real_wrist_best.json`) has, per camera:

```json
{
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist": [k1, k2, p1, p2, k3],
  "T_tool0_from_optical": [[4x4 matrix]]
}
```

- **`K`** — the pinhole camera intrinsic matrix: focal length (`fx`, `fy`,
  in pixels) and principal point (`cx`, `cy`). Converts a 3-D point in the
  camera's own coordinate frame into a 2-D pixel, ignoring lens distortion.
- **`dist`** — 5 lens distortion coefficients (radial `k1,k2,k3` + tangential
  `p1,p2`), OpenCV's standard model. These correct for the real lens not
  being a perfect pinhole (this rig's lenses have meaningful barrel
  distortion, k1 ≈ -0.4).
- **`T_tool0_from_optical`** — a 4×4 rigid transform from that camera's own
  optical frame into one shared **common frame**. In this file the common
  frame is chosen to be `left`'s own optical frame (so `left`'s own
  `T_tool0_from_optical` is just the identity matrix) — the field name is
  kept as `T_tool0_from_optical` only for schema consistency with the
  simulation calibration, it does **not** mean the real robot's wrist frame.

All three cameras are rigidly bolted together (wrist-mounted), so this
common-frame transform is fixed and reusable across every image, regardless
of where the robot arm itself is pointed.

### Where these numbers come from
Charuco board captures (`~/ws_nr/calibration/three_camera_charuco_*`):
photograph a checkerboard-with-ArUco-markers pattern from all 3 cameras
simultaneously, many times, at different positions/angles. OpenCV's
calibration routines solve for the `K`/`dist` that best explains all the
observed corner detections (`cv2.calibrateCamera`), and the rigid
`left↔center`/`left↔right` transforms via stereo calibration
(`cv2.stereoCalibrate`) using views where the board is visible in both
cameras at once.

**This tool's current calibration** (`calibration.real_wrist_best.json`) is
a *robust joint refinement* combining three separate Charuco datasets (a
large 10×7 board, a "close" 6×4 board, and a "dense" 8×6 board), optimizing
one consistent 3-camera rig geometry against all of them at once with a
2-pixel robust rejection cutoff (175 of 300 total triplets kept). See §7 for
why this replaced two earlier (wrong) calibration attempts.

---

## 3. The labeling workflow, end to end

1. **Launch** (`python main.py`) → choose **SFP** or **SC** → choose a
   folder with `left/`, `center/`, `right/` subfolders of images.
2. **Triplet discovery**: the tool matches images across the 3 subfolders
   by filename stem (e.g. `frame_001.png` in all three) — that's one
   "triplet". A missing view in one camera is tolerated (must be skipped).
3. **Per triplet, per object** (Plug or Port), a **guided flow**:
   - **Port**: single pass, Left → Center → Right, every keypoint placed
     manually, no calibration involved at all.
   - **Plug**, 2 phases:
     - **Phase 1 (green)**: Left → Center → Right, placing *only* the
       reference-point set for each camera (see §1). Nothing else is shown.
     - **Auto-calc** runs once, automatically, the moment Phase 1 finishes
       across all 3 views (§4 + §5).
     - **Phase 2 (review)**: Left → Center → Right again, now showing every
       point (green + whatever got filled in) for the human to check and,
       if needed, click-to-override any calculated point.
4. Clicking a point always goes through **ROI zoom** if you use it (drag a
   box → Confirm ROI) for sub-pixel-accurate placement on the full-resolution
   image — critical at this scale, since these are ~5-13mm features on a
   2464×2056 image.
5. **Save**: every single point placement is written immediately and
   atomically to `<folder>/labels/<stem>.json`. Reopening the same folder
   resumes exactly where you left off (already-done/partial cameras are
   skipped or revisited automatically per phase).
6. **Export** (File → Export YOLO-pose labels): writes a ready-to-train
   YOLO-pose dataset from every camera image marked fully "done" for an
   object, using only `training_labels` (e.g. SFP plug exports a1-a8 only —
   a9-a12 exist purely to help the math, they are not a training target).

---

## 4. Calculating a point without calibration: direct 2-D midpoints

For SFP plug, `a5 = midpoint(a1, a9)`, `a6 = midpoint(a2, a10)`, `a7 =
midpoint(a3, a11)`, `a8 = midpoint(a4, a12)` — this is a real, physical fact
about the connector (the mid-depth mark sits exactly halfway between the
tip and the 1.5in reference line).

**Wherever both parent points are already manually placed in the SAME
camera view**, the tool computes the midpoint the simplest possible way:
average the two parents' pixel coordinates directly, in that image. No
calibration, no 3-D triangulation, no camera-to-camera math — the result
can't inherit calibration error or cross-camera disagreement, because it
never leaves 2-D pixel space for that one image.

```
midpoint_x = (parent_a.x + parent_b.x) / 2
midpoint_y = (parent_a.y + parent_b.y) / 2
```

Displayed in **teal** (`midpoint_2d_point`) so it's visually distinct from a
calibrated result. This is preferred whenever it's available — it covers 8
of the 12 a5-a8 slots across a typical labeled triplet (left gets a6/a7/a8
this way, center gets a7/a8, right gets a5/a7/a8) — leaving only the labels
whose parent isn't visible/manual in that particular camera (a1, a2, a5, a6,
a9, a10, in the views where they're not a reference point) needing the
calibrated pipeline below.

**Caveat, stated honestly**: a 2-D pixel midpoint is not *exactly* the
reprojection of the true 3-D midpoint under perspective (only exact for
orthographic projection, or a line exactly perpendicular to the camera's
view direction). For a small object at typical working-distance standoff,
this error is normally much smaller than what the calibrated pipeline
itself was contributing — see §7 for the measured comparison.

---

## 5. Calculating a point WITH calibration: the multi-view pose pipeline

For any label that direct-midpoint can't cover, the tool falls back to
fitting the plug's full 3-D pose from whichever cameras have their complete
reference set placed, then reprojecting every label into every camera.

### Step 1 — per-camera pose solve (`geometry.py::_solve_pose_branches`)
For each camera with all its reference points placed, solve: given the
known 3-D positions of those reference points (`local_keypoints_m`) and
their observed 2-D pixels, and the camera's `K`/`dist`, find the rigid
transform `(R, t)` (rotation + translation) that maps the object's local
frame into that camera's optical frame.

The method used depends on the point count and geometry, dispatched
automatically:
- **Exactly 3 points** → P3P (`cv2.solveP3P`). If those 3 points are
  coplanar (they usually are — a face of a rigid connector), this has a
  genuine **2-fold mirror ambiguity**: two different 3-D poses both
  reproject perfectly onto the same 3 points. Both are returned as
  candidate "branches".
- **≥4 coplanar points** → IPPE (`cv2.solvePnPGeneric` with
  `SOLVEPNP_IPPE`), which has the same 2-fold ambiguity, explicitly
  returned as 2 branches.
- **≥4 non-coplanar points** → SQPNP, a single well-posed solution, no
  ambiguity (checked via an SVD-based coplanarity test on the actual
  local-keypoint set, not assumed).

SFP plug's left/right reference sets (6 points each) are **not coplanar**
(one corner sits at a different local `x` than the other 5, because a9-a12
were added specifically for this) — so left and right each resolve to one
unambiguous pose on their own. Center's 4-point set is still coplanar.

### Step 2 — cross-camera disambiguation
Every branch from every camera is converted into the shared frame (via
`T_tool0_from_optical`) and every (branch, branch) pair across two cameras
is compared:
```
position_disagreement_mm = |position_A - position_B| * 1000
rotation_disagreement_deg = angle between rotation_A and rotation_B
```
The pair with the smallest disagreement is kept. A genuinely correct match
agrees to a few mm / ~1°; a mirror-ambiguity mismatch disagrees by tens of
mm / tens of degrees — the two cases are not subtle, this check reliably
tells them apart.

### Step 3 — fuse and reproject
The winning pair is averaged (translation: mean; rotation: mean quaternion)
into one final pose, which is then reprojected into every camera's own
frame to produce a pixel position for every remaining label:
```
pixel = K · distort( (R_cam_from_object · local_point) + t_cam_from_object )
```

### Step 4 — accept, warn, or flag
- `position_disagreement_mm <= 5.0` **and** `rotation_disagreement_deg <=
  5.0` → accepted, shown in **blue** (`calculated_point`).
- Either threshold exceeded → still used (nothing else to fall back to),
  but shown in **orange** (`warning_point`) so the human reviewer knows to
  check it carefully in Phase 2.
- Fewer than 2 cameras have their reference set complete yet → nothing is
  calculated, a status message says exactly which cameras are still needed.

A manual click on a calculated (blue/orange) point during Phase 2 is tagged
`manual_override` and is never silently recomputed over again.

---

## 6. Coordinate/units summary (so the numbers in config.json make sense)

- `local_keypoints_m`: **metres**, in the object's own local frame, origin
  and axis choice are arbitrary but must be internally consistent (SFP
  plug's origin is the tip face, `+z` extends into the connector body).
- Pixel coordinates everywhere else: the **original, full-resolution image**
  pixel grid (e.g. 2464×2056) — the canvas widget deliberately builds its
  scene 1:1 from the full pixmap, so a zoomed ROI view never changes what a
  click actually reports; only the on-screen magnification changes.
- `T_tool0_from_optical`: metres, in whatever frame `left`'s optical frame
  defines as the origin.

---

## 7. Worked example: the real debugging session that shaped this design

This section exists so a future "why does X work this way" question has a
concrete answer, not just an assertion.

**Step 1 — a first geometry guess turned out wrong, and was caught.** SFP
plug's cross-section was initially copied from the simulation model
(12.8×7.6mm). A real caliper measurement gave 12.7×5.715mm. Applying the
"corrected" number made the real-data fit *worse* (8.5mm→13.7mm
disagreement, same clicks) — a red flag investigated rather than ignored.

**Step 2 — ruled out calibration and camera identity before touching
geometry again.** Recomputed intrinsics and extrinsics directly from the
Charuco calibration's own raw corner detections and compared to the stored
values: matched to 6+ decimal places. Cross-checked camera serial numbers
in the real capture's own metadata against the calibration's camera
mapping: exact match, no swap. Neither was the problem.

**Step 3 — isolated the error with a single-camera test.** Solved each
camera's pose from *only its own* reference points (no cross-camera math
possible) and checked reprojection error on those same points. This showed
near-face points (a1-a4) fitting well, far points (a9-a12) fitting poorly,
on *both* independent cameras — ruling out click noise or a calibration
issue (neither would produce that specific, repeatable per-point pattern)
and pointing squarely at the geometry model.

**Step 4 — the real height was actually 0.275in, not 0.225in.** A second,
corrected caliper reading, verified the same way (isolated single-camera
residual, no cross-camera math): roughly halved the residual on both
cameras (left 6.74→4.29px, right 8.03→4.06px mean). Real, measured
improvement, not just taking a new number on faith.

**Step 5 — a completely separate calibration bug was found afterward.** The
user discovered the Charuco board size assumption used for the original
calibration was itself wrong, and produced a new, corrected calibration by
jointly refining three board datasets. Before adopting it: independently
re-derived one of the three source datasets from its own raw corner data
(88.256mm baseline) and confirmed it matched the new calibration
(88.848mm) to under 1mm — the old calibration's baseline (96.14mm) was
off by a real, consistent ~8%. Switching to the corrected calibration, with
the *same* real clicks and the *same* corrected geometry, dropped the
cross-camera disagreement from 9.57mm to 2.68mm.

**Net effect**: two independent, real problems (a plug-thickness
measurement error, and a calibration board-scale error) were both
contributing to the same symptom at the same time, and were only
distinguishable by isolating each variable and verifying with real
computation at every step — never by accepting a plausible-sounding
explanation on its own. Residual ~4-6px per point on a9-a12 specifically
remains, and is still suspected to be a real housing taper/step at that
depth rather than a further cross-section correction — an open question for
future investigation, honestly stated as such rather than papered over.

---

## 8. Practical playbook: diagnosing a bad auto-calc result

If a calculated point looks visibly wrong, this is the order that actually
worked, cheapest/most-isolating checks first:

1. **Check the status label** — did it say "pending" (not enough
   reference views yet) or report a disagreement number? A `warning_point`
   (orange) already tells you the pipeline itself flagged it.
2. **Re-derive the calibration from its own raw source data** (if you have
   Charuco `metadata/*.json` corner detections) and diff against the
   calibration file in use — this rules out a stale/corrupted/wrong
   calibration file in minutes, and is the *first* thing to check, not the
   last, since it's the cheapest to fully rule out.
3. **Confirm camera identity** by serial number, not assumption — real
   capture metadata usually states which physical camera (serial) produced
   each `left`/`center`/`right` image; compare against the calibration's
   own per-camera serials.
4. **Solve each camera's pose independently from only its own points** and
   check reprojection error on those same points. This can't be affected by
   calibration extrinsics or cross-camera math at all — a bad result here
   is either click imprecision or a genuinely wrong local geometry model,
   never a calibration bug.
5. **Compare direct 2-D midpoints against calibrated reprojections** for
   the same label, where both are computable — a large gap (tens to
   hundreds of pixels) points at the calibrated pipeline specifically; a
   small gap (a few pixels) is consistent with ordinary perspective/noise.
6. Only after 2-5 come back clean should a geometry re-measurement be
   trusted as *the* fix — and even then, verify it the same way (isolated
   single-camera residual, before vs after) rather than accepting a new
   caliper reading on faith.
