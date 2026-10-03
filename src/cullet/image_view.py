from PySide6.QtCore import QElapsedTimer, QEvent, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

MIN_ZOOM = 0.02
MAX_ZOOM = 32.0
WHEEL_ZOOM_STEP = 1.25
WHEEL_NOTCH = 120  # angleDelta units of one mouse wheel notch
PAN_STEP_DIVISOR = 5  # a single pan() call from a script moves this fraction of the viewport
PAN_SPEED_PX_PER_S = 1500  # while an arrow key is held
PAN_TICK_MS = 16
PAN_KEYS = {
    Qt.Key.Key_Left: (-1, 0),
    Qt.Key.Key_Right: (1, 0),
    Qt.Key.Key_Up: (0, -1),
    Qt.Key.Key_Down: (0, 1),
}


class WheelAccumulator:
    """Turns wheel deltas into whole notches, so touchpads with many small deltas step once."""

    def __init__(self):
        self.accum = 0

    def steps(self, delta: int) -> int:
        self.accum += delta
        steps = int(self.accum / WHEEL_NOTCH)
        self.accum -= steps * WHEEL_NOTCH
        return steps


class ImageView(QGraphicsView):
    """
    Shows one image. In fit mode the image is shrunk to the window (never enlarged) and refit
    on resize; any manual zoom leaves fit mode.

    Below 1:1 the full image is not what gets painted. Qt filters a scaled pixmap from the four
    nearest source pixels, so fitting a 4000x3000 photo into a window looks at barely a quarter
    of it and the result aliases: fine detail turns into moire and the picture reads as soft.
    Instead the image is halved as often as the zoom allows (a mip pyramid, every level a
    smooth average of the one before) and only the remaining factor, always between 1 and 2,
    is left to Qt. One halving of a 12 MP image costs about 13 ms against 100 ms for its
    decode, and levels are built on demand and thrown away with the image.

    Above 1:1 there is nothing to filter: interpolating a magnified pixel only smears its
    edges, so the pixels stay square unless the smooth filter is switched on.
    """

    zoom_changed = Signal(float)
    wheel_navigate = Signal(int)  # +1 per wheel notch down (next), -1 per notch up (prev)
    double_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wheel_accum = WheelAccumulator()
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.fit_mode = True
        self.sharp_zoom = True
        self._source = QImage()
        self._mips: list[QImage] = []
        self._level = -1

        self.setBackgroundBrush(QColor("#202020"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # arrow keys pan continuously while held, driven by a timer instead of key auto-repeat
        self._held_pan_keys: set[Qt.Key] = set()
        self._pan_timer = QTimer(self)
        self._pan_timer.setInterval(PAN_TICK_MS)
        self._pan_timer.timeout.connect(self._pan_tick)
        self._pan_clock = QElapsedTimer()
        self._pan_remainder = [0.0, 0.0]

    def has_image(self) -> bool:
        return not self._source.isNull()

    def image_size(self) -> tuple[int, int]:
        """The size of the image itself, not of the mip level currently painted."""
        return self._source.width(), self._source.height()

    def set_image(self, qimage: QImage) -> None:
        self._source = qimage
        self._mips = [qimage]
        self._level = -1
        self._scene.setSceneRect(0, 0, qimage.width(), qimage.height())
        if self.fit_mode:
            self.fit()
        else:
            self._apply_level()
            self.zoom_changed.emit(self.zoom())

    def clear(self) -> None:
        self._source = QImage()
        self._mips = []
        self._level = -1
        self._item.setPixmap(QPixmap())
        self._item.setTransform(QTransform())
        self._scene.setSceneRect(0, 0, 0, 0)
        self.zoom_changed.emit(self.zoom())

    def mip_level(self) -> int:
        """How often the painted image was halved, 0 for the image itself."""
        return self._level

    def set_sharp_zoom(self, sharp: bool) -> None:
        """With sharp zoom a magnified pixel is a square, otherwise Qt interpolates it."""
        self.sharp_zoom = sharp
        self._apply_level()
        self.viewport().update()

    def _mip(self, level: int) -> QImage:
        while len(self._mips) <= level:
            previous = self._mips[-1]
            self._mips.append(
                previous.scaled(
                    max(1, previous.width() // 2),
                    max(1, previous.height() // 2),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        return self._mips[level]

    def _apply_level(self) -> None:
        """Paint from the smallest mip level that is still at least as large as the image on
        screen. The item is scaled back up to the source size, so the scene keeps source pixel
        coordinates and zoom, panning and the status bar do not know about any of this."""
        if not self.has_image():
            return
        # a high dpi screen paints more device pixels than the zoom alone suggests
        scale = self.zoom() * self.devicePixelRatioF()
        nearest = self.sharp_zoom and scale > 1
        self._item.setTransformationMode(
            Qt.TransformationMode.FastTransformation
            if nearest
            else Qt.TransformationMode.SmoothTransformation
        )
        level = 0
        while scale * 2 ** (level + 1) <= 1 and min(self.image_size()) >> (level + 1) >= 1:
            level += 1
        if level == self._level:
            return
        self._level = level
        image = self._mip(level)
        self._item.setPixmap(QPixmap.fromImage(image))
        self._item.setTransform(
            QTransform().scale(
                self._source.width() / image.width(), self._source.height() / image.height()
            )
        )

    def zoom(self) -> float:
        return self.transform().m11()

    def fit(self) -> None:
        self.fit_mode = True
        self.resetTransform()
        if self.has_image():
            img_w, img_h = self.image_size()
            view = self.viewport().size()
            scale = min(1.0, view.width() / img_w, view.height() / img_h)
            self.scale(scale, scale)
            self._apply_level()
            self.centerOn(self._item)
        self.zoom_changed.emit(self.zoom())

    def zoom_actual(self) -> None:
        self.fit_mode = False
        self.resetTransform()
        self._apply_level()
        self.zoom_changed.emit(self.zoom())

    def zoom_by(self, factor: float) -> None:
        self.fit_mode = False
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom() * factor))
        factor = new_zoom / self.zoom()
        self.scale(factor, factor)
        self._apply_level()
        self.zoom_changed.emit(self.zoom())

    def pan(self, dx: int, dy: int) -> None:
        """Scroll by pixels. Does nothing when the whole image is visible."""
        h_bar, v_bar = self.horizontalScrollBar(), self.verticalScrollBar()
        h_bar.setValue(h_bar.value() + dx)
        v_bar.setValue(v_bar.value() + dy)

    def pan_step(self) -> tuple[int, int]:
        size = self.viewport().size()
        return size.width() // PAN_STEP_DIVISOR, size.height() // PAN_STEP_DIVISOR

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self.double_clicked.emit()
        event.accept()

    # ---------- continuous keyboard panning

    def event(self, event: QEvent) -> bool:
        # claim the arrow keys before the window shortcuts see them, so press and release
        # arrive here and the pan can run while the key is held
        if event.type() == QEvent.Type.ShortcutOverride and event.key() in PAN_KEYS:
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key not in PAN_KEYS:
            super().keyPressEvent(event)
            return
        event.accept()
        if event.isAutoRepeat():
            return
        self._held_pan_keys.add(key)
        if not self._pan_timer.isActive():
            self._pan_remainder = [0.0, 0.0]
            self._pan_clock.start()
            self._pan_timer.start()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key not in PAN_KEYS:
            super().keyReleaseEvent(event)
            return
        event.accept()
        if event.isAutoRepeat():
            return
        self._held_pan_keys.discard(key)
        if not self._held_pan_keys:
            self._pan_timer.stop()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        # key releases are lost when the focus moves, e.g. to a dialog
        self._held_pan_keys.clear()
        self._pan_timer.stop()
        super().focusOutEvent(event)

    def _pan_tick(self) -> None:
        seconds = self._pan_clock.restart() / 1000
        direction_x = sum(PAN_KEYS[key][0] for key in self._held_pan_keys)
        direction_y = sum(PAN_KEYS[key][1] for key in self._held_pan_keys)
        self._pan_remainder[0] += direction_x * PAN_SPEED_PX_PER_S * seconds
        self._pan_remainder[1] += direction_y * PAN_SPEED_PX_PER_S * seconds
        dx, dy = int(self._pan_remainder[0]), int(self._pan_remainder[1])
        self._pan_remainder[0] -= dx
        self._pan_remainder[1] -= dy
        self.pan(dx, dy)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        event.accept()
        if delta == 0:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(WHEEL_ZOOM_STEP if delta > 0 else 1 / WHEEL_ZOOM_STEP)
            return
        steps = self._wheel_accum.steps(delta)
        if steps != 0:
            self.wheel_navigate.emit(-steps)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit()
