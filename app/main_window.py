from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QFileDialog, QFrame, QGroupBox, QHBoxLayout,
                                 QLabel, QMainWindow, QMessageBox, QPushButton,
                                 QScrollArea, QSizePolicy, QSplitter,
                                 QVBoxLayout, QWidget)

from . import state as st
from .canvas import ImageCanvas
from .config import AppConfig, load_calibration
from .geometry import (auto_calculate, compute_direct_midpoints,
                        compute_parallelogram_completion, triangulate_and_reproject)
from .exporter import export_route_object

REF_GREEN = "reference"
REF_CALC = "calculated"
REF_WARN = "warning"


class KeypointButton(QPushButton):
    def __init__(self, label: str, index: int):
        super().__init__(label)
        self.label = label
        self.index = index
        self.setCheckable(True)
        self.setMinimumHeight(34)
        self.setFont(QFont("", 11, QFont.Bold))


class MainWindow(QMainWindow):
    def __init__(self, route: str, folder: Path, config: AppConfig,
                 calibration: Optional[Dict] = None):
        super().__init__()
        self.route = route
        self.folder = folder
        self.config = config
        self.calibration = calibration or {}
        self.setWindowTitle(f"AT Labeling Tool -- {route.upper()}  [{folder}]")
        self._fit_window_to_screen()

        self.triplets: List[st.TripletState] = st.load_or_init_states(folder, route)
        self.idx = st.first_unresolved_index(self.triplets)

        self.current_camera: Optional[str] = None
        self.current_object: Optional[str] = None
        self.armed_label: Optional[str] = None
        self.keypoint_buttons: Dict[str, KeypointButton] = {}
        self._undo_stack: List[tuple] = []  # (camera, object, label)

        # Guided flow state. For Plug objects this is a 2-phase loop over
        # the 3 cameras -- "green" (place only the reference points on
        # left/center/right in turn) then "review" (revisit left/center/
        # right once more, now showing the auto-calculated blue points for
        # confirmation/override). Port objects use a single "single" phase
        # since there is no reference/calculated split for them.
        self.flow_object: Optional[str] = None
        self.flow_stage: Optional[str] = None  # "green" | "review" | "single"
        self.flow_order = ["left", "center", "right"]
        self.flow_pos = 0

        self._build_ui()
        self._build_shortcuts()
        self._goto_triplet_menu()

    # ------------------------------------------------------------------
    def _fit_window_to_screen(self):
        """Size and position the window to always fit entirely inside the
        current screen's available geometry (never taller/wider than the
        display, never off the edge) -- no fixed pixel size assumption."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        margin = 40
        max_w, max_h = avail.width() - margin, avail.height() - margin
        # Prefer 1500x950, but never exceed the screen; the "comfortable
        # minimum" (900x600) is itself capped at the screen size too, so it
        # can never push the window past the display on a small screen.
        width = min(1500, max_w)
        width = max(width, min(900, max_w))
        height = min(950, max_h)
        height = max(height, min(600, max_h))
        self.resize(width, height)
        self.move(
            avail.x() + (avail.width() - self.width()) // 2,
            avail.y() + (avail.height() - self.height()) // 2,
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        self.progress_label = QLabel()
        self.progress_label.setFont(QFont("", 12, QFont.Bold))
        outer.addWidget(self.progress_label)

        self.calibration_banner = QLabel()
        self.calibration_banner.setFont(QFont("", 10, QFont.Bold))
        if self.calibration:
            self.calibration_banner.setText(
                f"Calibration LOADED ({len(self.calibration)} cameras: {', '.join(self.calibration.keys())}) "
                f"-- Plug auto-calc active once >=2 views' green points are placed."
            )
            self.calibration_banner.setStyleSheet("color: white; background-color: #166534; padding: 4px;")
        else:
            self.calibration_banner.setText(
                "NO CALIBRATION LOADED -- all points (including Plug) must be placed manually. "
                "Restart and load a calibration.json to enable auto-calc."
            )
            self.calibration_banner.setStyleSheet("color: white; background-color: #991b1b; padding: 4px;")
        outer.addWidget(self.calibration_banner)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter, 1)

        # left panel -- inside a scroll area so a long keypoint-button list
        # (or a small screen) never forces the window taller than the
        # display; it scrolls internally instead. ------------------------
        left_container = QWidget()
        self.left_layout = QVBoxLayout(left_container)
        self.left_layout.setAlignment(Qt.AlignTop)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        # right panel: canvas + ROI toolbar -------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        toolbar = QHBoxLayout()
        self.btn_roi_select = QPushButton("Select ROI")
        self.btn_roi_confirm = QPushButton("Confirm ROI")
        self.btn_roi_reset = QPushButton("Reset ROI (full image)")
        self.btn_roi_select.clicked.connect(self._on_roi_select)
        self.btn_roi_confirm.clicked.connect(self._on_roi_confirm)
        self.btn_roi_reset.clicked.connect(self._on_roi_reset)
        for b in (self.btn_roi_select, self.btn_roi_confirm, self.btn_roi_reset):
            toolbar.addWidget(b)
        toolbar.addStretch(1)
        self.status_label = QLabel("")
        toolbar.addWidget(self.status_label)
        right_layout.addLayout(toolbar)

        self.canvas = ImageCanvas()
        self.canvas.point_placed.connect(self._on_point_placed)
        right_layout.addWidget(self.canvas, 1)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        export_action = QAction("&Export YOLO-pose labels...", self)
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)
        change_folder_action = QAction("&Open different folder...", self)
        change_folder_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(change_folder_action)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._on_undo)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self._on_enter_pressed)
        QShortcut(QKeySequence(Qt.Key_S), self, activated=self._on_skip_pressed)
        for i in range(1, 9):
            QShortcut(QKeySequence(str(i)), self, activated=lambda i=i: self._arm_by_index(i - 1))

    def _clear_left_panel(self):
        while self.left_layout.count():
            item = self.left_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.keypoint_buttons.clear()

    # ------------------------------------------------------------------
    # Triplet-level menu: choose Plug or Port (guided flow across all 3
    # camera views), or navigate/skip the whole triplet.
    # ------------------------------------------------------------------
    def _goto_triplet_menu(self):
        self.current_camera = None
        self.current_object = None
        self.flow_object = None
        self.flow_stage = None
        self.armed_label = None
        self.canvas.arm_placement(False)
        self._clear_left_panel()

        triplet = self.triplets[self.idx]
        self.progress_label.setText(
            f"Triplet {self.idx + 1}/{len(self.triplets)}: {triplet.stem}   "
            f"({'resolved' if triplet.is_fully_resolved() else 'in progress'})"
        )

        title = QLabel(f"Route: {self.route.upper()}")
        title.setFont(QFont("", 13, QFont.Bold))
        self.left_layout.addWidget(title)

        cam_status = QLabel(self._camera_status_summary(triplet))
        cam_status.setWordWrap(True)
        self.left_layout.addWidget(cam_status)

        box = QGroupBox("Label")
        vbox = QVBoxLayout(box)
        for obj in ("plug", "port"):
            statuses = [triplet.cameras[cam][obj]["status"] for cam in self.flow_order]
            n_done = sum(1 for s in statuses if s in (st.STATUS_DONE, st.STATUS_SKIPPED))
            fully_resolved = n_done == 3
            btn = QPushButton(f"{obj.capitalize()}  ({n_done}/3 views resolved)")
            btn.setMinimumHeight(50)
            btn.setFont(QFont("", 12, QFont.Bold))
            btn.setEnabled(not fully_resolved)
            btn.clicked.connect(lambda _, o=obj: self._start_flow(o))
            vbox.addWidget(btn)
        self.left_layout.addWidget(box)

        nav_box = QHBoxLayout()
        prev_btn = QPushButton("< Prev triplet")
        prev_btn.clicked.connect(self._prev_triplet)
        prev_btn.setEnabled(self.idx > 0)
        next_btn = QPushButton("Next triplet >")
        next_btn.clicked.connect(self._next_triplet)
        next_btn.setEnabled(self.idx < len(self.triplets) - 1)
        nav_box.addWidget(prev_btn)
        nav_box.addWidget(next_btn)
        self.left_layout.addLayout(nav_box)

        skip_triplet_btn = QPushButton("Skip entire triplet")
        skip_triplet_btn.clicked.connect(self._skip_triplet)
        self.left_layout.addWidget(skip_triplet_btn)

        self.canvas._scene.clear()
        self.status_label.setText("")

    def _camera_status_summary(self, triplet: st.TripletState) -> str:
        lines = []
        for cam in self.flow_order:
            cs = triplet.cameras[cam]
            missing = not triplet.image_paths.get(cam)
            bits = f"plug:{cs['plug']['status']}  port:{cs['port']['status']}"
            lines.append(f"{cam}: {'MISSING IMAGE' if missing else bits}")
        return "\n".join(lines)

    def _prev_triplet(self):
        if self.idx > 0:
            self.idx -= 1
            self._goto_triplet_menu()

    def _next_triplet(self):
        if self.idx < len(self.triplets) - 1:
            self.idx += 1
            self._goto_triplet_menu()
        else:
            QMessageBox.information(self, "Done", "This is the last triplet.")

    def _skip_triplet(self):
        triplet = self.triplets[self.idx]
        for cam in self.flow_order:
            triplet.mark_camera_status(cam, st.STATUS_SKIPPED)
            triplet.mark_object_status(cam, "plug", st.STATUS_SKIPPED)
            triplet.mark_object_status(cam, "port", st.STATUS_SKIPPED)
        self._next_triplet()

    # ------------------------------------------------------------------
    # Guided flow: for "plug" this is 2 phases (green, then review), each a
    # left -> center -> right pass; for "port" it is a single manual pass.
    # ------------------------------------------------------------------
    def _start_flow(self, obj_type: str):
        self.flow_object = obj_type
        cfg = self.config.object_config(self.route, obj_type)
        self.flow_stage = "green" if cfg["reference_points"] else "single"
        self.flow_pos = 0
        self._advance_to_next_flow_step(entering=True)

    def _camera_needs_visit(self, camera: str, obj_type: str, stage: str) -> bool:
        triplet = self.triplets[self.idx]
        status = triplet.cameras[camera][obj_type]["status"]
        if status == st.STATUS_SKIPPED:
            return False
        if stage == "green":
            # Green already placed (partial/done) -> nothing to do in this phase.
            return status == st.STATUS_NOT_STARTED
        # review / single: only fully-done views can be skipped over.
        return status != st.STATUS_DONE

    def _advance_to_next_flow_step(self, entering: bool = False):
        """Move flow_pos forward, skipping cameras that don't need this
        stage, until we land on one that does or run out -- in which case
        the stage (or the whole flow) is complete."""
        if not entering:
            self.flow_pos += 1
        while self.flow_pos < len(self.flow_order):
            cam = self.flow_order[self.flow_pos]
            missing_image = not self.triplets[self.idx].image_paths.get(cam)
            if missing_image:
                self.triplets[self.idx].mark_object_status(cam, self.flow_object, st.STATUS_SKIPPED)
                self.flow_pos += 1
                continue
            if self._camera_needs_visit(cam, self.flow_object, self.flow_stage):
                break
            self.flow_pos += 1
        else:
            self._flow_stage_complete()
            return
        self._enter_flow_step()

    def _flow_stage_complete(self):
        if self.flow_stage == "green":
            self._try_auto_calculate(self.flow_object)
            self.flow_stage = "review"
            self.flow_pos = 0
            self._advance_to_next_flow_step(entering=True)
        else:
            # "review" or "single" stage finished -> this object is done
            # for the whole triplet. Back to the Plug/Port menu.
            self.flow_object = None
            self.flow_stage = None
            self._goto_triplet_menu()
            triplet = self.triplets[self.idx]
            if triplet.is_fully_resolved() and self.idx < len(self.triplets) - 1:
                self._next_triplet()

    def _enter_flow_step(self):
        self.current_camera = self.flow_order[self.flow_pos]
        self.current_object = self.flow_object
        self._goto_labeling(self.flow_object)

    # ------------------------------------------------------------------
    # Labeling UI for one (camera, object) step of the guided flow.
    # ------------------------------------------------------------------
    def _goto_labeling(self, obj_type: str):
        self.armed_label = None
        self._clear_left_panel()

        cfg = self.config.object_config(self.route, obj_type)
        labels = cfg["labels"]
        ref_cfg = cfg["reference_points"]
        camera = self.current_camera
        stage = self.flow_stage
        required_here = ref_cfg[camera] if ref_cfg else labels

        triplet = self.triplets[self.idx]
        img_path = triplet.image_paths.get(camera, "")
        if img_path:
            self.canvas.load_image(img_path)

        cam_num = self.flow_order.index(camera) + 1
        if stage == "green":
            phase_text = f"Phase 1/2 -- REFERENCE POINTS ONLY"
        elif stage == "review":
            phase_text = f"Phase 2/2 -- REVIEW / COMPLETE ALL POINTS"
        else:
            phase_text = "Label all points"
        title = QLabel(f"{self.route.upper()} {obj_type.capitalize()}\n"
                        f"{phase_text}\n{camera.upper()}  ({cam_num}/3)")
        title.setFont(QFont("", 12, QFont.Bold))
        self.left_layout.addWidget(title)

        ref_img_path = self.config.reference_image_path(self.route, obj_type, camera)
        ref_label = QLabel()
        if ref_img_path.is_file():
            pix = QPixmap(str(ref_img_path))
            ref_label.setPixmap(pix.scaledToWidth(360, Qt.SmoothTransformation))
        else:
            ref_label.setText("(reference image missing)")
        ref_label.setFrameShape(QFrame.Box)
        self.left_layout.addWidget(ref_label)

        if ref_cfg and stage == "green":
            hint = QLabel(f"Place these {len(required_here)} reference points, in any order: "
                           f"{', '.join(required_here)}.\nThe other points are calculated "
                           f"automatically once left+right (and center) reference points are all in --"
                           f" you'll review/adjust them in Phase 2.")
        elif ref_cfg and stage == "review":
            if self.calibration:
                hint = QLabel("Green = your reference points (still editable). "
                               "Blue = auto-calculated -- click a blue button to override it manually. "
                               "Orange = auto-calc disagreement warning, check it carefully.")
            else:
                hint = QLabel("No calibration loaded -- nothing was auto-calculated. "
                               "Place every remaining (blue) point manually.")
        else:
            hint = QLabel("Place every point manually for this object.")
        hint.setWordWrap(True)
        self.left_layout.addWidget(hint)

        btn_box = QGroupBox("Keypoints")
        grid = QVBoxLayout(btn_box)
        points = triplet.get_points(camera, obj_type)
        visible_labels = required_here if (ref_cfg and stage == "green") else labels
        for i, label in enumerate(labels):
            if label not in visible_labels:
                continue
            btn = KeypointButton(label, i)
            is_ref = (not ref_cfg) or (label in required_here)
            placed = label in points
            self._style_keypoint_button(btn, is_ref, placed, points.get(label, {}).get("source"))
            btn.clicked.connect(lambda _, l=label: self._arm_label(l))
            grid.addWidget(btn)
            self.keypoint_buttons[label] = btn
        self.left_layout.addWidget(btn_box)

        action_row = QHBoxLayout()
        undo_btn = QPushButton("Undo last (Ctrl+Z)")
        undo_btn.clicked.connect(self._on_undo)
        action_row.addWidget(undo_btn)
        clear_btn = QPushButton("Clear armed point")
        clear_btn.clicked.connect(self._clear_armed_point)
        action_row.addWidget(clear_btn)
        self.left_layout.addLayout(action_row)

        skip_btn = QPushButton(f"Skip {obj_type} for this image")
        skip_btn.clicked.connect(self._skip_object)
        self.left_layout.addWidget(skip_btn)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._finish_object)
        self.left_layout.addWidget(self.next_btn)

        back_btn = QPushButton("< Back to Plug/Port menu")
        back_btn.clicked.connect(self._goto_triplet_menu)
        self.left_layout.addWidget(back_btn)

        self._redraw_points()
        self._update_next_enabled()

    def _style_keypoint_button(self, btn: KeypointButton, is_ref: bool, placed: bool, source: Optional[str]):
        if source == REF_WARN:
            color = self.config.color("warning_point")
        elif is_ref:
            color = self.config.color("reference_point")
        elif source == "midpoint_2d":
            color = self.config.color("midpoint_2d_point")
        elif source == "parallelogram_2d":
            color = self.config.color("parallelogram_2d_point")
        elif source == "triangulated":
            color = self.config.color("triangulated_point")
        else:
            color = self.config.color("calculated_point")
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: white; border-radius: 4px; }}"
            f"QPushButton:checked {{ border: 3px solid {self.config.color('active_button')}; }}"
        )
        btn.setText(f"{btn.label}" + ("  ✓" if placed else ""))

    def _redraw_points(self):
        self.canvas.clear_all_points()
        camera = self.current_camera
        obj_type = self.current_object
        cfg = self.config.object_config(self.route, obj_type)
        ref_cfg = cfg["reference_points"]
        required_here = ref_cfg[camera] if ref_cfg else cfg["labels"]
        triplet = self.triplets[self.idx]
        points = triplet.get_points(camera, obj_type)
        for label, p in points.items():
            is_ref = (not ref_cfg) or (label in required_here)
            source = p.get("source")
            if source == REF_WARN:
                color = self.config.color("warning_point")
            elif is_ref:
                color = self.config.color("reference_point")
            elif source == "midpoint_2d":
                color = self.config.color("midpoint_2d_point")
            elif source == "parallelogram_2d":
                color = self.config.color("parallelogram_2d_point")
            elif source == "triangulated":
                color = self.config.color("triangulated_point")
            else:
                color = self.config.color("calculated_point")
            self.canvas.set_point(label, p["x"], p["y"], color)

    # ------------------------------------------------------------------
    def _arm_label(self, label: str):
        for lbl, btn in self.keypoint_buttons.items():
            btn.setChecked(lbl == label)
        self.armed_label = label
        self.canvas.arm_placement(True)

    def _arm_by_index(self, index: int):
        if self.current_object is None:
            return
        labels = self.config.labels(self.route, self.current_object)
        if 0 <= index < len(labels) and labels[index] in self.keypoint_buttons:
            self._arm_label(labels[index])

    def _clear_armed_point(self):
        if not self.armed_label or self.current_object is None:
            return
        triplet = self.triplets[self.idx]
        triplet.clear_point(self.current_camera, self.current_object, self.armed_label)
        self._goto_labeling(self.current_object)

    def _on_undo(self):
        if not self._undo_stack:
            return
        camera, obj_type, label = self._undo_stack.pop()
        triplet = self.triplets[self.idx]
        triplet.clear_point(camera, obj_type, label)
        if camera == self.current_camera and obj_type == self.current_object:
            self._goto_labeling(self.current_object)

    def _on_point_placed(self, x: float, y: float):
        if not self.armed_label or self.current_object is None:
            return
        camera, obj_type, label = self.current_camera, self.current_object, self.armed_label
        triplet = self.triplets[self.idx]
        cfg = self.config.object_config(self.route, obj_type)
        ref_cfg = cfg["reference_points"]
        required_here = ref_cfg[camera] if ref_cfg else cfg["labels"]
        # A manual click on a required (green) point is just "manual". A
        # manual click on a non-reference (blue) point -- only possible
        # during the "review" phase -- is an explicit override of whatever
        # auto-calc put there, and must never be silently recomputed away.
        source = "manual" if (not ref_cfg or label in required_here) else "manual_override"
        triplet.set_point(camera, obj_type, label, x, y, source)
        self._undo_stack.append((camera, obj_type, label))
        self._goto_labeling(obj_type)
        # NOTE: auto-calc is intentionally NOT triggered here. It runs
        # exactly once, explicitly, when the whole "green" phase finishes
        # across all 3 views (see _flow_stage_complete) -- not per-point
        # placement, which would race ahead and finish the object before
        # the user ever reaches the review phase they asked to see.

    def _try_auto_calculate(self, obj_type: str):
        cfg = self.config.object_config(self.route, obj_type)
        ref_cfg = cfg["reference_points"]
        labels = cfg["labels"]
        local_kps = self.config.local_keypoints(self.route, obj_type)
        midpoint_of = self.config.midpoint_of(self.route, obj_type)
        triplet = self.triplets[self.idx]

        points_by_camera = {}
        for cam in ("left", "center", "right"):
            pts = triplet.get_points(cam, obj_type)
            cam_pts = {l: (p["x"], p["y"]) for l, p in pts.items()}
            if cam_pts:
                points_by_camera[cam] = cam_pts

        # Step 1: fill anything computable as a direct 2-D pixel midpoint,
        # per camera, with no calibration and no 3-D triangulation -- this
        # can't inherit calibration/cross-camera-agreement error, so prefer
        # it wherever both parent points are already placed in the same view.
        direct_filled_msgs = []
        if midpoint_of:
            direct = compute_direct_midpoints(midpoint_of, points_by_camera)
            for cam, cam_points in direct.items():
                for label, (x, y) in cam_points.items():
                    triplet.set_point(cam, obj_type, label, x, y, "midpoint_2d")
                    points_by_camera.setdefault(cam, {})[label] = (x, y)
                direct_filled_msgs.append(f"{cam}:{sorted(cam_points.keys())}")
        if direct_filled_msgs:
            self.status_label.setText("Direct 2-D midpoint filled -- " + ", ".join(direct_filled_msgs))

        # Step 1b: parallelogram-completion, per camera -- also calibration-
        # free. Runs after the midpoint pass (above) so a rectangle missing
        # its 4th corner can use a corner the midpoint pass JUST filled in
        # (e.g. the mid-depth rectangle's a5 needs a6/a7/a8, which midpoint
        # only just computed from a2/a10, a3/a11, a4/a12).
        rectangles = self.config.rectangles(self.route, obj_type)
        parallelogram_filled_msgs = []
        if rectangles:
            completed = compute_parallelogram_completion(rectangles, points_by_camera)
            for cam, cam_points in completed.items():
                for label, (x, y) in cam_points.items():
                    triplet.set_point(cam, obj_type, label, x, y, "parallelogram_2d")
                    points_by_camera.setdefault(cam, {})[label] = (x, y)
                parallelogram_filled_msgs.append(f"{cam}:{sorted(cam_points.keys())}")
        if parallelogram_filled_msgs:
            prefix = (self.status_label.text() + "  ") if direct_filled_msgs else ""
            self.status_label.setText(prefix + "Parallelogram-completed -- " + ", ".join(parallelogram_filled_msgs))

        # Step 1c: direct two-view triangulation + reprojection. Preferred
        # over the rigid-pose-fit fallback (step 3) whenever it applies: if
        # a camera is still missing a label, but the OTHER TWO cameras both
        # already have it (typically because steps 1/1b just filled them in,
        # calibration-free), triangulate that point directly from those two
        # real observations and reproject it into the missing camera. This
        # doesn't assume the object's exact rigid shape is correct -- it only
        # needs real point correspondence + calibration -- so it's simpler
        # and more robust than fitting the whole local_keypoints_m model to
        # a reference subset.
        triangulate_filled_msgs = []
        if self.calibration:
            cams = ("left", "center", "right")
            for target_cam in cams:
                source_cams = tuple(c for c in cams if c != target_cam)
                still_missing = [l for l in labels if l not in points_by_camera.get(target_cam, {})]
                if not still_missing:
                    continue
                filled = triangulate_and_reproject(points_by_camera, self.calibration, source_cams, target_cam)
                new_labels = []
                for label in still_missing:
                    if label not in filled:
                        continue
                    x, y = filled[label]
                    triplet.set_point(target_cam, obj_type, label, x, y, "triangulated")
                    points_by_camera.setdefault(target_cam, {})[label] = (x, y)
                    new_labels.append(label)
                if new_labels:
                    triangulate_filled_msgs.append(f"{target_cam}:{sorted(new_labels)}")
        if triangulate_filled_msgs:
            prefix = self.status_label.text() + "  " if (direct_filled_msgs or parallelogram_filled_msgs) else ""
            self.status_label.setText(prefix + "Triangulated from other 2 views -- " + ", ".join(triangulate_filled_msgs))

        # Step 2: only labels NOT covered by steps 1/1b/1c need the rigid-
        # pose-fit calibrated pipeline at all (e.g. a point that's still
        # missing in >=2 cameras at once, so there's no pair to triangulate
        # from and no reference set complete enough to solve a pose either).
        still_needed_any = any(
            label not in points_by_camera.get(cam, {})
            for cam in ("left", "center", "right")
            for label in labels
            if label not in ref_cfg.get(cam, [])
        )
        if not still_needed_any:
            if self.current_object == obj_type:
                self._goto_labeling(obj_type)
            return

        if not self.calibration:
            self.status_label.setText(
                (self.status_label.text() + "  " if direct_filled_msgs else "")
                + "No calibration loaded -- remaining points must be placed manually."
            )
            return

        result = auto_calculate(local_kps, labels, ref_cfg, points_by_camera, self.calibration)
        if not result.success:
            solvable = [c for c, refs in ref_cfg.items() if len(refs) >= 3]
            done_solvable = [c for c in solvable
                              if all(l in points_by_camera.get(c, {}) for l in ref_cfg[c])]
            still_needed = [c for c in solvable if c not in done_solvable]
            self.status_label.setText(
                f"Auto-calc pending -- needs green points on >=2 of {solvable} "
                f"(done: {done_solvable or 'none'}; still needed: {still_needed})"
            )
            return

        source = REF_WARN if result.warning else REF_CALC
        msg = (f"Auto-calc OK using {result.used_cameras}: "
               f"{result.position_disagreement_mm:.2f}mm / {result.rotation_disagreement_deg:.2f}deg")
        if result.warning:
            msg = "WARNING (check!) " + msg
        self.status_label.setText(msg)

        for cam, pts in result.points_by_camera.items():
            required_cam = ref_cfg.get(cam, [])
            existing = triplet.get_points(cam, obj_type)
            for i, label in enumerate(labels):
                if label in required_cam:
                    continue  # never overwrite a manually-placed reference point
                x, y = float(pts[i][0]), float(pts[i][1])
                # Never overwrite a manual override, or a value already
                # filled by the (preferred, calibration-free) direct 2-D
                # midpoint pass above.
                if existing.get(label, {}).get("source") in ("manual_override", "midpoint_2d", "parallelogram_2d", "triangulated"):
                    continue
                triplet.set_point(cam, obj_type, label, x, y, source)
            # Deliberately NOT auto-promoting "partial" -> "done" here: the
            # guided flow's whole point is a mandatory review pass over the
            # newly-calculated blue points before an object counts as done
            # for a camera -- auto-completing it here would silently skip
            # that check the user explicitly asked for.

        if self.current_object == obj_type:
            self._goto_labeling(obj_type)

    def _required_labels_for_next(self, camera: str, obj_type: str):
        """What must be placed before you can move on. During the "green"
        phase, only the reference set is required (blue points come later,
        once >=2 views' greens are in). During "review"/"single", every
        label is required."""
        cfg = self.config.object_config(self.route, obj_type)
        ref_cfg = cfg["reference_points"]
        if ref_cfg and self.flow_stage == "green":
            return ref_cfg[camera]
        return cfg["labels"]

    def _update_next_enabled(self):
        obj_type = self.current_object
        camera = self.current_camera
        triplet = self.triplets[self.idx]
        points = triplet.get_points(camera, obj_type)
        required = self._required_labels_for_next(camera, obj_type)
        self.next_btn.setEnabled(all(l in points for l in required))

    def _finish_object(self):
        camera, obj_type = self.current_camera, self.current_object
        triplet = self.triplets[self.idx]
        cfg = self.config.object_config(self.route, obj_type)
        labels = cfg["labels"]
        points = triplet.get_points(camera, obj_type)
        required = self._required_labels_for_next(camera, obj_type)
        if not all(l in points for l in required):
            QMessageBox.warning(self, "Incomplete", "Required keypoints for this step are not all placed yet.")
            return
        fully_placed = all(l in points for l in labels)
        triplet.mark_object_status(camera, obj_type, st.STATUS_DONE if fully_placed else st.STATUS_PARTIAL)
        self._advance_to_next_flow_step()

    def _skip_object(self):
        camera, obj_type = self.current_camera, self.current_object
        triplet = self.triplets[self.idx]
        triplet.mark_object_status(camera, obj_type, st.STATUS_SKIPPED)
        self._advance_to_next_flow_step()

    def _on_enter_pressed(self):
        if self.current_object is not None and self.next_btn.isEnabled():
            self._finish_object()

    def _on_skip_pressed(self):
        if self.current_object is not None:
            self._skip_object()

    # ------------------------------------------------------------------
    def _on_roi_select(self):
        self.canvas.begin_roi_selection()

    def _on_roi_confirm(self):
        self.canvas.confirm_roi()

    def _on_roi_reset(self):
        self.canvas.reset_roi()

    # ------------------------------------------------------------------
    def _on_export(self):
        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder", str(self.folder))
        if not out_dir:
            return
        counts = {}
        for obj_type in ("plug", "port"):
            n = export_route_object(self.folder, self.route, obj_type, self.config, Path(out_dir))
            counts[obj_type] = n
        QMessageBox.information(self, "Export complete",
                                  "\n".join(f"{k}: {v} labeled images" for k, v in counts.items()))

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with left/center/right subfolders")
        if not folder:
            return
        self.folder = Path(folder)
        self.triplets = st.load_or_init_states(self.folder, self.route)
        self.idx = st.first_unresolved_index(self.triplets)
        self._goto_triplet_menu()
