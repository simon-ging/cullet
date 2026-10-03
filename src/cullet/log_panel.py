import logging
from datetime import datetime

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QPlainTextEdit, QWidget

logger = logging.getLogger(__name__)
MAX_LINES = 2000


class LogPanel(QPlainTextEdit):
    """Read-only list of what the viewer did to files, one timestamped line per operation."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(MAX_LINES)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))

    def log(self, message: str) -> None:
        self.appendPlainText(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}")
        logger.info(message)
