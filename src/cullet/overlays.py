"""
Overlays drawn on top of the image view: a centered help box and subtitle-style info text in
the bottom corners. Both ignore the mouse, so dragging the image underneath still works.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics, QPainter, QPaintEvent
from PySide6.QtWidgets import QLabel, QWidget

MARGIN = 12
INFO_POINT_SIZE = 14
OUTLINE_PX = 2
# offsets of the black copies drawn behind the white text, gives a readable outline anywhere
OUTLINE_OFFSETS = [
    (dx, dy)
    for dx in range(-OUTLINE_PX, OUTLINE_PX + 1)
    for dy in range(-OUTLINE_PX, OUTLINE_PX + 1)
    if (dx, dy) != (0, 0)
]


class OutlinedLabel(QLabel):
    """White text with a black outline, like a subtitle."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        font = QFont()
        font.setPointSize(INFO_POINT_SIZE)
        font.setBold(True)
        self.setFont(font)
        self.setContentsMargins(OUTLINE_PX, OUTLINE_PX, OUTLINE_PX, OUTLINE_PX)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setFont(self.font())
        rect = self.contentsRect()
        flags = int(self.alignment())
        painter.setPen(Qt.GlobalColor.black)
        for dx, dy in OUTLINE_OFFSETS:
            painter.drawText(rect.translated(dx, dy), flags, self.text())
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(rect, flags, self.text())


class Overlays:
    def __init__(self, parent: QWidget):
        self.parent = parent

        self.help_box = QLabel(parent)
        self.help_box.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.help_box.setTextFormat(Qt.TextFormat.PlainText)
        self.help_box.setStyleSheet(
            "QLabel { background: white; color: black; border: 1px solid black; padding: 8px; }"
        )
        self.help_box.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.help_box.hide()

        self.info_left = OutlinedLabel(parent)
        self.info_right = OutlinedLabel(parent)
        self.info_top_right = OutlinedLabel(parent)
        self.info_top_left = OutlinedLabel(parent)
        for label in self.info_labels():
            label.hide()
        self._info_left_text = ""
        self._info_top_left_text = ""

    def info_labels(self) -> tuple[OutlinedLabel, ...]:
        return self.info_left, self.info_right, self.info_top_right, self.info_top_left

    def set_help_text(self, text: str) -> None:
        self.help_box.setText(text)
        self.relayout()

    def toggle_help(self) -> None:
        self.help_box.setVisible(not self.help_box.isVisible())
        self.relayout()

    def set_info(self, left: str, right: str, top_right: str = "", top_left: str = "") -> None:
        """The left text may hold several lines, e.g. the folder and below it the group."""
        self._info_left_text = left
        self._info_top_left_text = top_left
        self.info_right.setText(right)
        self.info_top_right.setText(top_right)
        self.relayout()

    def toggle_info(self) -> None:
        visible = not self.info_left.isVisible()
        for label in self.info_labels():
            label.setVisible(visible)
        self.relayout()

    def relayout(self) -> None:
        pw, ph = self.parent.width(), self.parent.height()
        self.info_right.adjustSize()
        self.info_top_right.adjustSize()
        metrics = QFontMetrics(self.info_left.font())
        # shorten the folder path from the left so it never runs under the filename
        available = pw - self.info_right.width() - 3 * MARGIN
        self.info_left.setText(
            "\n".join(
                metrics.elidedText(line, Qt.TextElideMode.ElideLeft, available)
                for line in self._info_left_text.split("\n")
            )
        )
        # the tags share their row with the exif line, and give way to it
        self.info_top_left.setText(
            metrics.elidedText(
                self._info_top_left_text,
                Qt.TextElideMode.ElideRight,
                pw - self.info_top_right.width() - 3 * MARGIN,
            )
        )
        for label in (self.help_box, *self.info_labels()):
            label.adjustSize()
            label.raise_()
        self.help_box.move((pw - self.help_box.width()) // 2, (ph - self.help_box.height()) // 2)
        self.info_left.move(MARGIN, ph - self.info_left.height() - MARGIN)
        self.info_right.move(
            pw - self.info_right.width() - MARGIN, ph - self.info_right.height() - MARGIN
        )
        self.info_top_right.move(pw - self.info_top_right.width() - MARGIN, MARGIN)
        self.info_top_left.move(MARGIN, MARGIN)
