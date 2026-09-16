#!/usr/bin/env python3
"""AT Labeling Tool -- entry point.

Cross-platform (Linux/Windows/Mac) keypoint labeling tool for SFP/SC plug
and port images, with reference-point-driven auto-calculation of the
remaining keypoints when a real per-camera calibration is supplied.

Run: python main.py
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import AppConfig, load_calibration
from app.dialogs import ModeDialog, choose_calibration, choose_folder
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    mode_dialog = ModeDialog()
    if mode_dialog.exec() != QDialog.Accepted or not mode_dialog.route:
        return 0
    route = mode_dialog.route

    folder = choose_folder()
    if folder is None:
        return 0

    config = AppConfig()

    calibration = {}
    reply = QMessageBox.question(
        None, "Calibration",
        "Load a camera calibration file now?\n\n"
        "With calibration: reference (green) points auto-calculate the "
        "remaining (blue) keypoints for Plug objects.\n"
        "Without calibration: all points must be placed manually "
        "(same as Port labeling).",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
    )
    if reply == QMessageBox.Yes:
        calib_path = choose_calibration()
        if calib_path and calib_path.is_file():
            try:
                calibration = load_calibration(calib_path)
            except Exception as exc:
                QMessageBox.warning(None, "Calibration failed to load", str(exc))
                calibration = {}

    window = MainWindow(route=route, folder=folder, config=config, calibration=calibration)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
