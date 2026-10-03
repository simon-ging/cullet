from datetime import datetime
from pathlib import Path

from natsort import natsorted
from PySide6.QtCore import QAbstractListModel, QDir, QModelIndex, QObject, Qt
from PySide6.QtWidgets import QFileSystemModel

from cullet.image_io import IMAGE_SUFFIXES, read_exif_datetime

# "exif" sorts by the date taken and falls back to the mtime for files without one
SORT_ORDERS = ["name", "mtime", "exif"]


def list_images(folder: Path, recursive: bool = False, sort: str = "name") -> list[Path]:
    """Image files of the folder. Recursive listing skips hidden directories. Ties (and the
    name order) are natural sort of the relative path."""
    if recursive:
        candidates = [
            p
            for p in folder.rglob("*")
            if not any(part.startswith(".") for part in p.relative_to(folder).parts[:-1])
        ]
    else:
        candidates = list(folder.iterdir())
    files = [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    files = natsorted(files, key=lambda p: p.relative_to(folder).as_posix())
    if sort == "name":
        return files
    if sort == "mtime":
        return sorted(files, key=lambda p: p.stat().st_mtime)
    if sort == "exif":
        return sorted(files, key=_exif_or_mtime_key)
    raise ValueError(f"Unknown sort order {sort}, expected one of {SORT_ORDERS}")


def _exif_or_mtime_key(path: Path) -> str:
    date = read_exif_datetime(path)
    if date is not None:
        return date
    return f"{datetime.fromtimestamp(path.stat().st_mtime):%Y:%m:%d %H:%M:%S}"


class FileListModel(QAbstractListModel):
    """The image files of one folder, or any list of files with their own row labels."""

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.files: list[Path] = []
        self.labels: list[str] | None = None

    def set_folder(self, folder: Path, recursive: bool = False, sort: str = "name") -> None:
        self.set_files(list_images(folder, recursive, sort), folder)

    def set_files(self, files: list[Path], folder: Path, labels: list[str] | None = None) -> None:
        """Show these files. Rows show the path relative to the folder unless labels are given."""
        assert labels is None or len(labels) == len(files), f"{len(labels)} vs {len(files)}"
        self.beginResetModel()
        self.folder = folder
        self.files = files
        self.labels = labels
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.files)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        path = self.files[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            if self.labels is not None:
                return self.labels[index.row()]
            return path.relative_to(self.folder).as_posix()
        if role == Qt.ItemDataRole.ToolTipRole:
            return path.as_posix()
        return None


def make_folder_tree_model(parent: QObject | None = None) -> QFileSystemModel:
    model = QFileSystemModel(parent)
    model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
    model.setRootPath(QDir.rootPath())
    return model
