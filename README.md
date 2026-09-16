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
4. For each triplet: pick a camera view (Left/Center/Right), then Plug or
   Port. Reference image for that route/object/camera is shown at the top
   of the left panel. Click a keypoint button to "arm" it, then click the
   point's location on the image (right panel) to place it. Click again to
   replace a mistaken placement.
5. **ROI zoom**: "Select ROI" then drag a box on the image, "Confirm ROI" to
   zoom into it (label sub-pixel-accurately), "Reset ROI" to see the whole
   image again.
6. For **Plug** objects with a calibration loaded: place the green
   (reference) points first. Once >=2 camera views have their full
   reference set placed, the remaining blue points are auto-calculated and
   drawn -- click a blue button to manually override any of them.
7. **Skip** is available at every level (a whole triplet, one camera image,
   or just Plug/Port for that image) -- real data has occlusions and
   missing views.
8. Progress saves after every single point placement
   (`<folder>/labels/<stem>.json`, atomic writes). Re-opening the same
   folder resumes exactly where you left off; a fresh folder starts clean.
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
for real camera captures at 2464x2056. Supply your own real calibration file
(camera intrinsic calibration + hand-eye/rig extrinsics) for real hardware.
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
