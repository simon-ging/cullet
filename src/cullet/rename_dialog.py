from PySide6.QtCore import QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QVBoxLayout, QWidget

from cullet.file_ops import propose_rename


class RenameDialog(QDialog):
    """Rename with the cursor placed for adding tags, see file_ops.propose_rename."""

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Rename")
        self.setMinimumWidth(700)
        text, self.cursor = propose_rename(name)
        self.edit = QLineEdit(text)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.edit)
        layout.addWidget(buttons)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        # QLineEdit selects all its text when it receives the initial focus, so place the
        # cursor after that has happened
        QTimer.singleShot(0, self.place_cursor)

    def place_cursor(self) -> None:
        self.edit.setFocus()
        self.edit.deselect()
        self.edit.setCursorPosition(self.cursor)

    def new_name(self) -> str:
        return self.edit.text().strip()
