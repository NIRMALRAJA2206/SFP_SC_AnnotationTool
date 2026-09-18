"""Zoomable, ROI-able image canvas built on QGraphicsView.

Coordinate discipline (per design review): the scene is built 1:1 from the
full-resolution pixmap, so scene coordinates ARE original-image pixel
coordinates. Every click is mapped view -> scene via mapToScene and used
directly; nothing is ever computed in view/screen pixels. Zooming (fitInView
on an ROI rect, or resetting to the whole image) only changes the view
transform, never the scene content or the point data.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QCursor, QFont, QImage, QPainter,
                            QPen, QPixmap)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsPixmapItem,
                                 QGraphicsRectItem, QGraphicsScene,
                                 QGraphicsSimpleTextItem, QGraphicsView,
                                 QRubberBand)


POINT_RADIUS = 7.0


class ImageCanvas(QGraphicsView):
    point_placed = Signal(float, float)  # scene (= original image pixel) coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._image_rect = QRectF()
        self._point_items: Dict[str, Tuple[QGraphicsEllipseItem, QGraphicsSimpleTextItem]] = {}
        self._image_path: Optional[str] = None

        self._roi_mode = False
        self._roi_origin: Optional[QPointF] = None
        self._roi_rubber: Optional[QRubberBand] = None
        self._roi_rect_scene: Optional[QRectF] = None

        self._place_armed = False

    # ---- image loading -----------------------------------------------
    def load_image(self, path: str) -> bool:
        if path == self._image_path and self._pixmap_item is not None:
            return True  # same image already loaded -- avoid a needless
            # scene teardown/rebuild on every single point placement
            # (_goto_labeling calls this unconditionally on every redraw).
        img = QImage(path)
        if img.isNull():
            return False
        # Tear down the OLD image/points via controlled removeItem calls
        # (matches _remove_point_item's pattern) rather than
        # QGraphicsScene.clear() -- clear() force-deletes the underlying
        # C++ objects immediately, which then double-frees when our own
        # _point_items dict (still holding Python references to those same
        # items) is cleared right after: this was the "free(): invalid
        # pointer / Aborted (core dumped)" crash on placing a point.
        self.clear_all_points()
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        pix = QPixmap.fromImage(img)
        self._pixmap_item = self._scene.addPixmap(pix)
        self._image_rect = QRectF(0, 0, pix.width(), pix.height())
        self._scene.setSceneRect(self._image_rect)
        self._image_path = path
        self.reset_roi()
        return True

    # ---- ROI -----------------------------------------------------------
    def begin_roi_selection(self):
        self._roi_mode = True
        self.setCursor(Qt.CrossCursor)

    def cancel_roi_selection(self):
        self._roi_mode = False
        if self._roi_rubber:
            self._roi_rubber.hide()
        self.unsetCursor()

    def reset_roi(self):
        self._roi_rect_scene = None
        if self._image_rect.isValid():
            self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def confirm_roi(self):
        if self._roi_rect_scene is not None and self._roi_rect_scene.width() > 4 and self._roi_rect_scene.height() > 4:
            self.fitInView(self._roi_rect_scene, Qt.KeepAspectRatio)
        self._roi_mode = False
        if self._roi_rubber:
            self._roi_rubber.hide()
        self.unsetCursor()

    # ---- point placement -------------------------------------------------
    def arm_placement(self, armed: bool):
        self._place_armed = armed
        self.setCursor(Qt.CrossCursor if armed else Qt.ArrowCursor)

    def set_point(self, label: str, x: float, y: float, color: str, filled: bool = True):
        self._remove_point_item(label)
        pen = QPen(QColor("black"))
        pen.setWidth(1)
        brush = QBrush(QColor(color)) if filled else QBrush(Qt.NoBrush)
        ellipse = QGraphicsEllipseItem(x - POINT_RADIUS, y - POINT_RADIUS, POINT_RADIUS * 2, POINT_RADIUS * 2)
        ellipse.setPen(pen)
        ellipse.setBrush(brush)
        ellipse.setZValue(10)
        self._scene.addItem(ellipse)

        text = QGraphicsSimpleTextItem(label)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        text.setFont(font)
        text.setBrush(QBrush(QColor("white")))
        text.setPen(QPen(QColor("black"), 0.6))
        text.setPos(x + POINT_RADIUS + 2, y - POINT_RADIUS - 8)
        text.setZValue(11)
        self._scene.addItem(text)

        self._point_items[label] = (ellipse, text)

    def remove_point(self, label: str):
        self._remove_point_item(label)

    def _remove_point_item(self, label: str):
        item = self._point_items.pop(label, None)
        if item:
            for it in item:
                self._scene.removeItem(it)

    def clear_all_points(self):
        for label in list(self._point_items.keys()):
            self._remove_point_item(label)

    # ---- mouse handling --------------------------------------------------
    def mousePressEvent(self, event):
        if self._roi_mode and event.button() == Qt.LeftButton:
            self._roi_origin = event.position().toPoint()
            if self._roi_rubber is None:
                self._roi_rubber = QRubberBand(QRubberBand.Rectangle, self)
            self._roi_rubber.setGeometry(self._roi_origin.x(), self._roi_origin.y(), 0, 0)
            self._roi_rubber.show()
            return
        if self._place_armed and event.button() == Qt.LeftButton:
            scene_pt = self.mapToScene(event.position().toPoint())
            if self._image_rect.contains(scene_pt):
                self.point_placed.emit(scene_pt.x(), scene_pt.y())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._roi_mode and self._roi_origin is not None and self._roi_rubber is not None:
            rect = self._normalized_rect(self._roi_origin, event.position().toPoint())
            self._roi_rubber.setGeometry(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._roi_mode and self._roi_origin is not None and event.button() == Qt.LeftButton:
            rect = self._normalized_rect(self._roi_origin, event.position().toPoint())
            top_left = self.mapToScene(rect.topLeft())
            bottom_right = self.mapToScene(rect.bottomRight())
            self._roi_rect_scene = QRectF(top_left, bottom_right).normalized()
            self._roi_origin = None
            return
        super().mouseReleaseEvent(event)

    @staticmethod
    def _normalized_rect(p1, p2):
        from PySide6.QtCore import QRect
        x0, y0 = min(p1.x(), p2.x()), min(p1.y(), p2.y())
        x1, y1 = max(p1.x(), p2.x()), max(p1.y(), p2.y())
        return QRect(x0, y0, x1 - x0, y1 - y0)
