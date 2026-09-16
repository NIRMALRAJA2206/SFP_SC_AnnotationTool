from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
                                 QMessageBox, QPushButton, QVBoxLayout)


class ModeDialog(QDialog):
    """Startup dialog: choose SFP or SC. Shown every launch, per requirement."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AT Labeling Tool")
        self.setMinimumWidth(360)
        self.route: Optional[str] = None

        layout = QVBoxLayout(self)
        title = QLabel("What are you going to label?")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        row = QHBoxLayout()
        sfp_btn = QPushButton("SFP")
        sc_btn = QPushButton("SC")
        for b in (sfp_btn, sc_btn):
            b.setMinimumHeight(60)
            b.setFont(QFont("", 13, QFont.Bold))
        sfp_btn.clicked.connect(lambda: self._choose("sfp"))
        sc_btn.clicked.connect(lambda: self._choose("sc"))
        row.addWidget(sfp_btn)
        row.addWidget(sc_btn)
        layout.addLayout(row)

    def _choose(self, route: str):
        self.route = route
        self.accept()


def choose_folder(parent=None) -> Optional[Path]:
    """Folder picker; validates that left/center/right subfolders exist
    (warns but still allows -- discover_triplets tolerates missing views)."""
    folder = QFileDialog.getExistingDirectory(
        parent, "Select folder containing left/center/right subfolders"
    )
    if not folder:
        return None
    path = Path(folder)
    missing = [c for c in ("left", "center", "right") if not (path / c).is_dir()]
    if missing:
        QMessageBox.warning(
            parent, "Missing camera subfolders",
            f"This folder is missing: {', '.join(missing)}. "
            f"Those views will be treated as absent and must be skipped per triplet."
        )
    return path


def choose_calibration(parent=None) -> Optional[Path]:
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select calibration.json (optional -- Cancel to skip)",
        str(Path(__file__).resolve().parent.parent), "JSON files (*.json)"
    )
    return Path(path) if path else None
