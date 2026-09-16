from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QFileDialog, QFrame, QGroupBox, QHBoxLayout,
                                 QLabel, QMainWindow, QMessageBox, QPushButton,
                                 QScrollArea, QSizePolicy, QSplitter,
                                 QVBoxLayout, QWidget)

from . import state as st
from .canvas import ImageCanvas
from .config import AppConfig, load_calibration
from .geometry import auto_calculate
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
        self.resize(1500, 950)

        self.triplets: List[st.TripletState] = st.load_or_init_states(folder, route)
        self.idx = st.first_unresolved_index(self.triplets)

        self.current_camera: Optional[str] = None
        self.current_object: Optional[str] = None
        self.armed_label: Optional[str] = None
        self.keypoint_buttons: Dict[str, KeypointButton] = {}
        self._undo_stack: List[tuple] = []  # (camera, object, label)

        self._build_ui()
        self._build_shortcuts()
        self._goto_camera_menu()

    # ------------------------------------------------------------------
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

        # left panel -----------------------------------------------------
        left_container = QWidget()
        left_container.setMinimumWidth(320)
        left_container.setMaximumWidth(420)
        self.left_layout = QVBoxLayout(left_container)
        self.left_layout.setAlignment(Qt.AlignTop)
        splitter.addWidget(left_container)

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
    # Stage: choose camera (left/center/right) for the current triplet
    # ------------------------------------------------------------------
    def _goto_camera_menu(self):
        self.current_camera = None
        self.current_object = None
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

        box = QGroupBox("Select camera view")
        vbox = QVBoxLayout(box)
        for cam in ("left", "center", "right"):
            cs = triplet.cameras[cam]
            missing = not triplet.image_paths.get(cam)
            status_bits = []
            for obj in ("plug", "port"):
                status_bits.append(f"{obj}:{cs[obj]['status']}")
            btn = QPushButton(f"{cam.capitalize()}  [{', '.join(status_bits)}]" + ("  (missing image)" if missing else ""))
            btn.setMinimumHeight(40)
            if missing:
                btn.setEnabled(False)
            btn.clicked.connect(lambda _, c=cam: self._goto_object_menu(c))
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

    def _prev_triplet(self):
        if self.idx > 0:
            self.idx -= 1
            self._goto_camera_menu()

    def _next_triplet(self):
        if self.idx < len(self.triplets) - 1:
            self.idx += 1
            self._goto_camera_menu()
        else:
            QMessageBox.information(self, "Done", "This is the last triplet.")

    def _skip_triplet(self):
        triplet = self.triplets[self.idx]
        for cam in ("left", "center", "right"):
            triplet.mark_camera_status(cam, st.STATUS_SKIPPED)
            triplet.mark_object_status(cam, "plug", st.STATUS_SKIPPED)
            triplet.mark_object_status(cam, "port", st.STATUS_SKIPPED)
        self._next_triplet()

    # ------------------------------------------------------------------
    # Stage: choose Plug / Port for the chosen camera
    # ------------------------------------------------------------------
    def _goto_object_menu(self, camera: str):
        self.current_camera = camera
        self.current_object = None
        self.armed_label = None
        self.canvas.arm_placement(False)
        self._clear_left_panel()

        triplet = self.triplets[self.idx]
        img_path = triplet.image_paths.get(camera, "")
        if img_path:
            self.canvas.load_image(img_path)
        self.status_label.setText(f"{camera} image loaded" if img_path else "no image")

        title = QLabel(f"{camera.capitalize()} image -- choose object to label")
        title.setFont(QFont("", 13, QFont.Bold))
        self.left_layout.addWidget(title)

        cs = triplet.cameras[camera]
        for obj in ("plug", "port"):
            done = cs[obj]["status"] in (st.STATUS_DONE, st.STATUS_SKIPPED)
            btn = QPushButton(f"{obj.capitalize()}  [{cs[obj]['status']}]")
            btn.setMinimumHeight(44)
            btn.setEnabled(not done)
            btn.clicked.connect(lambda _, o=obj: self._goto_labeling(o))
            self.left_layout.addWidget(btn)

        skip_btn = QPushButton("Skip this camera image entirely")
        skip_btn.clicked.connect(self._skip_camera)
        self.left_layout.addWidget(skip_btn)

        back_btn = QPushButton("< Back")
        back_btn.clicked.connect(self._goto_camera_menu)
        self.left_layout.addWidget(back_btn)

        if all(cs[o]["status"] in (st.STATUS_DONE, st.STATUS_SKIPPED) for o in ("plug", "port")):
            next_btn = QPushButton("Both done -- continue")
            next_btn.clicked.connect(self._advance_after_camera)
            self.left_layout.addWidget(next_btn)

    def _skip_camera(self):
        triplet = self.triplets[self.idx]
        triplet.mark_object_status(self.current_camera, "plug", st.STATUS_SKIPPED)
        triplet.mark_object_status(self.current_camera, "port", st.STATUS_SKIPPED)
        triplet.mark_camera_status(self.current_camera, st.STATUS_SKIPPED)
        self._advance_after_camera()

    def _advance_after_camera(self):
        order = ["left", "center", "right"]
        i = order.index(self.current_camera)
        if i + 1 < len(order):
            self._goto_object_menu(order[i + 1])
        else:
            self._goto_camera_menu()
            if self.idx < len(self.triplets) - 1 and self.triplets[self.idx].is_fully_resolved():
                self._next_triplet()

    # ------------------------------------------------------------------
    # Stage: labeling one object (plug or port) on one camera image
    # ------------------------------------------------------------------
    def _goto_labeling(self, obj_type: str):
        self.current_object = obj_type
        self.armed_label = None
        self._clear_left_panel()

        cfg = self.config.object_config(self.route, obj_type)
        labels = cfg["labels"]
        ref_cfg = cfg["reference_points"]
        camera = self.current_camera
        required_here = ref_cfg[camera] if ref_cfg else labels

        title = QLabel(f"{self.route.upper()} {obj_type.capitalize()} -- {camera}")
        title.setFont(QFont("", 13, QFont.Bold))
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

        if ref_cfg:
            hint = QLabel(f"Green = place manually first: {', '.join(required_here)}\n"
                           f"Blue = auto-calculated once >=2 views' greens are placed "
                           f"(click to override).")
        else:
            hint = QLabel("All points must be placed manually for this object.")
        hint.setWordWrap(True)
        self.left_layout.addWidget(hint)

        btn_box = QGroupBox("Keypoints")
        grid = QVBoxLayout(btn_box)
        triplet = self.triplets[self.idx]
        points = triplet.get_points(camera, obj_type)
        for i, label in enumerate(labels):
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

        back_btn = QPushButton("< Back")
        back_btn.clicked.connect(lambda: self._goto_object_menu(camera))
        self.left_layout.addWidget(back_btn)

        self._redraw_points()
        self._update_next_enabled()

    def _style_keypoint_button(self, btn: KeypointButton, is_ref: bool, placed: bool, source: Optional[str]):
        if source == REF_WARN:
            color = self.config.color("warning_point")
        elif is_ref:
            color = self.config.color("reference_point")
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
        if 0 <= index < len(labels):
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
        source = "manual"
        triplet.set_point(camera, obj_type, label, x, y, source)
        self._undo_stack.append((camera, obj_type, label))
        self._goto_labeling(obj_type)

        # try auto-calc if this completed the reference set for a plug-type object
        if ref_cfg and self.config.auto_calculate(self.route, obj_type):
            points = triplet.get_points(camera, obj_type)
            if all(l in points for l in required_here):
                self._try_auto_calculate(obj_type)

    def _try_auto_calculate(self, obj_type: str):
        cfg = self.config.object_config(self.route, obj_type)
        ref_cfg = cfg["reference_points"]
        labels = cfg["labels"]
        local_kps = self.config.local_keypoints(self.route, obj_type)
        triplet = self.triplets[self.idx]

        points_by_camera = {}
        for cam in ("left", "center", "right"):
            pts = triplet.get_points(cam, obj_type)
            cam_pts = {l: (p["x"], p["y"]) for l, p in pts.items()}
            if cam_pts:
                points_by_camera[cam] = cam_pts

        if not self.calibration:
            self.status_label.setText(
                "No calibration loaded -- blue points must be placed manually."
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
                # keep a manual override if the user already adjusted this blue point
                if existing.get(label, {}).get("source") == "manual_override":
                    continue
                triplet.set_point(cam, obj_type, label, x, y, source)

            # A camera left "partial" (green done, blue pending a 2nd view)
            # is now fully placed -- promote it to done automatically so the
            # user isn't forced to revisit and click Next again just to
            # acknowledge points that just got filled in.
            if triplet.cameras[cam][obj_type]["status"] == st.STATUS_PARTIAL:
                all_labels_now = triplet.get_points(cam, obj_type)
                if all(l in all_labels_now for l in labels):
                    triplet.mark_object_status(cam, obj_type, st.STATUS_DONE)

        if self.current_object == obj_type:
            self._goto_labeling(obj_type)

    def _required_labels_for_next(self, camera: str, obj_type: str):
        """What must be placed before you can move on: the reference (green)
        set for Plug objects (blue points may still be pending on a 2nd/3rd
        view before auto-calc can run), or every label for Port objects
        (nothing is auto-calculated there)."""
        cfg = self.config.object_config(self.route, obj_type)
        ref_cfg = cfg["reference_points"]
        if ref_cfg:
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
            QMessageBox.warning(self, "Incomplete",
                                  "Required (green) keypoints for this view are not all placed yet.")
            return
        fully_placed = all(l in points for l in labels)
        triplet.mark_object_status(camera, obj_type, st.STATUS_DONE if fully_placed else st.STATUS_PARTIAL)
        self._goto_object_menu(camera)

    def _skip_object(self):
        camera, obj_type = self.current_camera, self.current_object
        triplet = self.triplets[self.idx]
        triplet.mark_object_status(camera, obj_type, st.STATUS_SKIPPED)
        self._goto_object_menu(camera)

    def _on_enter_pressed(self):
        if self.current_object is not None and self.next_btn.isEnabled():
            self._finish_object()

    def _on_skip_pressed(self):
        if self.current_object is not None:
            self._skip_object()
        elif self.current_camera is not None:
            self._skip_camera()

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
        self._goto_camera_menu()
