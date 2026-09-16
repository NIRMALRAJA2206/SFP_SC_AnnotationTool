# AT Labeling Tool

Cross-platform (Linux / Windows / macOS) keypoint labeling tool for SFP and
SC plug/port images, built for triplet (left/center/right) camera captures.

## Install
```
pip install -r requirements.txt
python main.py
```
Tested with PySide6 6.11, OpenCV 5.0, NumPy 2.4, Python 3.12. Should work on
any recent Python 3.9+.

## Workflow
1. On launch: choose **SFP** or **SC** (asked every time you start the tool).
2. Choose the folder that contains `left/`, `center/`, `right/` subfolders
   of full-resolution images. Matching is by filename stem across the three
   subfolders (a "triplet"); a missing view in one camera is tolerated and
   must be explicitly skipped.
3. Optionally load a **calibration.json** (see below). Without one, all
   keypoints are placed manually (same as Port labeling always is).
4. For each triplet, pick **Plug** or **Port** once (not per camera) --
   the tool then guides you through a fixed sequence:
   - **Plug** (2 phases): **Phase 1** walks Left -> Center -> Right, showing
     only the green (reference) keypoint buttons -- place just those, in
     any order, then click Next to move to the next camera. Once all 3
     views' reference points are in, auto-calc runs once and **Phase 2**
     starts: Left -> Center -> Right again, now showing every point (green
     + the auto-calculated blue ones) so you can review and, if needed,
     click a blue button to manually override it before clicking Next.
   - **Port** (1 phase): Left -> Center -> Right, all keypoints manual (no
     green/blue split), click Next after each.
   Reference image for that route/object/camera is shown at the top of the
   left panel throughout. Click a keypoint button to "arm" it, then click
   the point's location on the image (right panel) to place it. Click again
   to replace a mistaken placement.
5. **ROI zoom**: "Select ROI" then drag a box on the image, "Confirm ROI" to
   zoom into it (label sub-pixel-accurately), "Reset ROI" to see the whole
   image again.
6. A camera finished in Phase 1 (green done, blue not yet calculable) is
   marked **partial** and is revisited automatically in Phase 2 -- it only
   becomes **done** once you click Next during the review phase.
7. **Skip** is available at every level (a whole triplet, or just Plug/Port
   for the camera image currently shown) -- real data has occlusions and
   missing views. A missing view in the triplet is skipped automatically.
8. Progress saves after every single point placement
   (`<folder>/labels/<stem>.json`, atomic writes). Re-opening the same
   folder resumes exactly where you left off -- already-done or partial
   cameras are skipped/revisited automatically within each phase; a fresh
   folder starts clean.
9. **File > Export YOLO-pose labels...** writes a ready-to-train dataset
   (`images/{train,val}`, `labels/{train,val}`, `dataset.yaml`, `manifest.json`)
   for every camera image marked "done" for that object.

## Keyboard shortcuts
- `1`-`8`: arm keypoint button N
- Click on image: place the armed point
- `Ctrl+Z`: undo the last placed point
- `Enter`: Next (once all keypoints for the current object are placed)
- `S`: Skip current object/camera

## calibration.json
Auto-calculation for Plug objects needs real per-camera intrinsics/extrinsics
for the physical rig that captured these images:
```json
{
  "image_size": {"width": 2464, "height": 2056},
  "cameras": {
    "left":   {"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "dist": [k1,k2,p1,p2,k3],
               "T_tool0_from_optical": [[4x4 row-major matrix]]},
    "center": {...},
    "right":  {...}
  }
}
```
`calibration.sim_example.json` in this repo is a **simulation-only** example
(the AIC Gazebo rig at 1152x1024) -- it documents the schema, it is NOT valid
for real camera captures at 2464x2056.

`calibration.real_wrist_v3.json` in this repo IS a real, usable calibration:
converted from `~/ws_nr/calibration/three_camera_charuco_v3` (Charuco board,
100 shared views, per-camera RMS ~0.7px, Basler acA2440-20gc + 8.5mm lens at
2464x2056 -- matches this tool's target resolution exactly). Camera mapping
used: `camera_1`(serial 25530362)`=left`, `camera_2`(25530348)`=center`,
`camera_3`(25530346)`=right`. The reference/common frame is `camera_1`
(left)'s own optical frame -- `T_tool0_from_optical` in this file really
means "T from the left camera's frame", the field name is kept only for
schema compatibility. **Re-run the Charuco calibration and regenerate this
file if focus, aperture, resolution, or camera mounting changes** (per that
calibration's own `capture_settings.json` note).

Without a calibration file, the tool still works -- Plug objects just fall
back to fully-manual labeling like Port objects already are.

### Why auto-calc needs >=2 views
Each camera's reference-point set (see `config.json`) is a set of coplanar
points, so a single camera's 3-point pose solve (P3P) has a genuine two-fold
mirror-branch ambiguity -- both solutions reproject perfectly onto that
camera's own 3 points. The tool resolves this by requiring a second camera's
independent reference-point solve and keeping whichever pair of branches
agrees with each other (in the shared `tool0` frame) -- a real match agrees
to a few mm / ~1 degree, a mirror mismatch disagrees by tens of mm/degrees.
If the best pair still disagrees by more than 5mm / 5deg, the calculated
points are drawn in a warning color instead of the normal blue.

## Configuring keypoints
`config.json` defines, per route (`sfp`/`sc`) and object (`plug`/`port`):
`labels`, `local_keypoints_m` (the object's real rigid-body geometry, only
needed for Plug auto-calc), `reference_points` (which labels are green per
camera), and `reference_image_dir` (thumbnail shown while labeling). Edit
this file to retarget the tool to a different connector or point layout.

## Project layout
```
ws_attool/
  main.py                 entry point
  config.json              keypoint/label/reference-point configuration
  calibration.sim_example.json   documented schema, simulation values only
  requirements.txt
  app/
    canvas.py              zoomable/ROI QGraphicsView image widget
    config.py               config + calibration loading
    dialogs.py              startup mode/folder/calibration dialogs
    exporter.py              YOLO-pose dataset export
    geometry.py              P3P + branch disambiguation + reprojection math
    main_window.py           main application window/state machine
    state.py                 resumable per-triplet JSON persistence
    yolo_format.py           standalone YOLO-pose label formatting
  assets/references/{sfp,sc}/{plug,port}/{left,center,right}.png
```
