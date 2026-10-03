"""
Thumbnails of the current folder: the model shared by the strip below the image and the grid
view, plus both views. Thumbnails are decoded in the background the first time a cell is
painted and kept in their own cache.
"""

from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)

from cullet.image_cache import ImageLoader
from cullet.image_view import WHEEL_NOTCH, WheelAccumulator

# thumbnails are decoded at this size and scaled down in the views, so cells up to this size
# are sharp and bigger ones slightly soft
THUMB_DECODE_SIZE = 768  # 256
CELL_SIZES = [64, 96, 128, 192, 256, 384, 512, 768]  # Ctrl+wheel steps through these
STRIP_DEFAULT_CELL_INDEX = 2
GRID_DEFAULT_CELL_INDEX = 3
CELL_PADDING = 6
SCROLLBAR_HEIGHT = 20
PAN_THRESHOLD_PX = 4
PLACEHOLDER_COLOR = QColor("#3a3a3a")


class ThumbnailModel(QAbstractListModel):
    def __init__(self, loader: ImageLoader, parent: QObject | None = None):
        super().__init__(parent)
        self.loader = loader
        self.files: list[Path] = []
        self._row_of: dict[str, int] = {}
        self.loader.image_ready.connect(self._on_image_ready)

    def set_files(self, files: list[Path]) -> None:
        self.beginResetModel()
        self.files = files
        self._row_of = {p.as_posix(): i for i, p in enumerate(files)}
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.files)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        path = self.files[index.row()]
        if role == Qt.ItemDataRole.DecorationRole:
            qimage = self.loader.get(path)
            if qimage is None:
                return None
            return QPixmap.fromImage(qimage)
        if role == Qt.ItemDataRole.ToolTipRole:
            return path.name
        return None

    def refresh(self, path: Path) -> None:
        """Decode again, e.g. after the file was rotated."""
        self.loader.invalidate(path)
        self._emit_changed(path.as_posix())

    def _on_image_ready(self, key: str) -> None:
        self._emit_changed(key)

    def _emit_changed(self, key: str) -> None:
        row = self._row_of.get(key)
        if row is None:
            return
        index = self.index(row)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])


class ThumbnailDelegate(QStyledItemDelegate):
    """
    Square cell of a fixed size, the thumbnail scaled to fit and centered. The size comes from
    the delegate and not from the data, because the view measures items before the
    thumbnails are decoded.
    """

    def __init__(self, cell_size: int, parent: QObject | None = None):
        super().__init__(parent)
        self.cell_size = cell_size

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(self.cell_size, self.cell_size)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        rect = option.rect
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(rect, option.palette.highlight())
        inner = rect.adjusted(CELL_PADDING, CELL_PADDING, -CELL_PADDING, -CELL_PADDING)
        pixmap = index.data(Qt.ItemDataRole.DecorationRole)
        if pixmap is None:
            painter.fillRect(inner, PLACEHOLDER_COLOR)
            return
        scaled = pixmap.scaled(
            inner.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = inner.x() + (inner.width() - scaled.width()) // 2
        y = inner.y() + (inner.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


class ThumbnailView(QListView):
    """
    Icon view with the shared mouse behaviour of strip and grid: wheel steps through the
    images like the single view, Ctrl+wheel changes the cell size, Shift+wheel and dragging
    scroll along the view's axis without changing the selection, a click selects.
    """

    wheel_navigate = Signal(int)
    # where the current cell ends up when the selection moves, see scroll_to_row
    SCROLL_HINT = QAbstractItemView.ScrollHint.EnsureVisible

    def __init__(self, cell_index: int, horizontal: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self._cell_index = cell_index
        self._horizontal = horizontal
        self.delegate = ThumbnailDelegate(self.cell_size(), self)
        self.setItemDelegate(self.delegate)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(not horizontal)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setUniformItemSizes(True)
        self.setGridSize(QSize(self.cell_size(), self.cell_size()))
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        # keys stay with the window shortcuts, the views are mouse-only
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._navigate_accum = WheelAccumulator()
        self._resize_accum = WheelAccumulator()
        self._press_pos: QPoint | None = None
        self._press_scroll = 0
        self._panning = False

    def scroll_to_row(self, index: QModelIndex) -> None:
        self.scrollTo(index, self.SCROLL_HINT)

    def cell_size(self) -> int:
        return CELL_SIZES[self._cell_index] + 2 * CELL_PADDING

    def change_cell_size(self, steps: int) -> None:
        new_index = min(len(CELL_SIZES) - 1, max(0, self._cell_index + steps))
        if new_index == self._cell_index:
            return
        self._cell_index = new_index
        current = self.currentIndex()
        self.delegate.cell_size = self.cell_size()
        self.setGridSize(QSize(self.cell_size(), self.cell_size()))
        self.on_cell_size_changed()
        self.doItemsLayout()
        if current.isValid():
            self.scroll_to_row(current)

    def on_cell_size_changed(self) -> None:
        pass

    def _scroll_bar(self):
        return self.horizontalScrollBar() if self._horizontal else self.verticalScrollBar()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        event.accept()
        if delta == 0:
            return
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.change_cell_size(self._resize_accum.steps(delta))
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            bar = self._scroll_bar()
            bar.setValue(bar.value() - delta * self.cell_size() // WHEEL_NOTCH)
            return
        steps = self._navigate_accum.steps(delta)
        if steps != 0:
            self.wheel_navigate.emit(-steps)

    # left button: drag pans, a click without drag selects. QAbstractItemView's own mouse
    # handling is bypassed, so there is no rubber band and no selection on drag.

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._press_pos = event.position().toPoint()
        self._press_scroll = self._scroll_bar().value()
        self._panning = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._press_pos is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position().toPoint() - self._press_pos
        if self._panning or abs(delta.y()) > PAN_THRESHOLD_PX or abs(delta.x()) > PAN_THRESHOLD_PX:
            self._panning = True
            along_axis = delta.x() if self._horizontal else delta.y()
            self._scroll_bar().setValue(self._press_scroll - along_axis)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._press_pos is None:
            super().mouseReleaseEvent(event)
            return
        if not self._panning:
            index = self.indexAt(event.position().toPoint())
            if index.isValid():
                self.setCurrentIndex(index)
                self.clicked.emit(index)
        self._press_pos = None
        self._panning = False


class ThumbnailStrip(ThumbnailView):
    """
    One row below the image, scrolls horizontally, as high as its cells.

    The current cell is kept in the middle of the strip instead of merely visible, so the
    images that come next are on screen before they are reached. Only the ends of the folder
    are off center, there is nothing to scroll to there.
    """

    SCROLL_HINT = QAbstractItemView.ScrollHint.PositionAtCenter

    def __init__(self, parent: QWidget | None = None):
        super().__init__(STRIP_DEFAULT_CELL_INDEX, horizontal=True, parent=parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.on_cell_size_changed()

    def on_cell_size_changed(self) -> None:
        self.setFixedHeight(self.cell_size() + SCROLLBAR_HEIGHT)


class ThumbnailGrid(ThumbnailView):
    """Wrapping grid instead of the image, scrolls vertically."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(GRID_DEFAULT_CELL_INDEX, horizontal=False, parent=parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # the grid holds the focus while it is shown, so the arrow key shortcuts of the view
        # stack apply to it
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def columns(self) -> int:
        """Items per row in the current layout, from the positions of the first row."""
        model = self.model()
        if model is None or model.rowCount() == 0:
            return 1
        first_y = self.visualRect(model.index(0)).y()
        count = 1
        while count < model.rowCount() and self.visualRect(model.index(count)).y() == first_y:
            count += 1
        return count
