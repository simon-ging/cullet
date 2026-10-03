"""
File operations of the viewer with an undo stack: move to the trash, rename, rotate.
Every operation is reported through the log callback in a short "VERB details" form.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from attrs import define
from PySide6.QtCore import QFile

from cullet.image_io import Transform, inverse, rotate_file

TAG_MARKER = "--tags--"


@define
class FileOp:
    kind: str  # DEL, MV or ROT
    src: Path
    dst: Path | None = None
    transform: Transform | None = None


class FileOps:
    def __init__(self, trash_dir: Path | None, log: Callable[[str], None]):
        """
        Args:
            trash_dir: deleted files are moved here. None uses the trash of the desktop, the
                one the file manager shows.
            log: called with one line per operation
        """
        self.trash_dir = None if trash_dir is None else Path(trash_dir)
        self.log = log
        self.undo_stack: list[FileOp] = []

    def trash(self, path: Path, note: str = "") -> Path:
        """Move the file into the trash. In a trash dir its absolute path is mirrored so nothing
        collides. The note is put into the log line, e.g. dimensions and similarity of a
        duplicate."""
        path = Path(path).absolute()
        if self.trash_dir is None:
            dst = move_to_system_trash(path)
        else:
            dst = free_path(self.trash_dir / path.relative_to(path.anchor))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, dst)
        self.undo_stack.append(FileOp("DEL", path, dst))
        self.log(f"DEL {note + ' ' if note else ''}{path.name} -> {dst}")
        return dst

    def move(self, path: Path, target_dir: Path) -> Path:
        """Move the file into the target dir, which is created if missing."""
        path = Path(path).absolute()
        dst = free_path(Path(target_dir).absolute() / path.name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, dst)
        self.undo_stack.append(FileOp("MV", path, dst))
        self.log(f"MV {path.name} -> {dst}")
        return dst

    def toggle_tag(self, path: Path, tag: str) -> Path:
        """Add the tag to the file name, or remove it if the name carries it already."""
        tags = tags_of(path.name)
        tags = [t for t in tags if t != tag] if tag in tags else tags + [tag]
        return self.rename(path, name_with_tags(path.name, tags))

    def rename(self, path: Path, new_name: str) -> Path:
        path = Path(path).absolute()
        dst = path.with_name(new_name)
        assert not dst.exists(), f"Cannot rename, {dst} exists already"
        path.rename(dst)
        self.undo_stack.append(FileOp("MV", path, dst))
        self.log(f"MV {path.name} -> {new_name}")
        return dst

    def rotate(self, path: Path, transform: Transform) -> None:
        path = Path(path).absolute()
        rotate_file(path, transform)
        self.undo_stack.append(FileOp("ROT", path, transform=transform))
        self.log(f"ROT {path.name} {transform.value}")

    def undo(self) -> Path | None:
        """Revert the last operation, return the path the file has afterwards."""
        if not self.undo_stack:
            self.log("UNDO nothing to undo")
            return None
        op = self.undo_stack.pop()
        if op.kind == "ROT":
            rotate_file(op.src, inverse(op.transform))
            self.log(f"UNDO ROT {op.src.name}")
            return op.src
        assert not op.src.exists(), f"Cannot undo, {op.src} exists again"
        if not op.dst.exists():
            raise FileNotFoundError(f"Cannot undo, {op.dst} is gone. Was the trash emptied?")
        op.src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(op.dst, op.src)
        if op.kind == "DEL" and self.trash_dir is None:
            # the system trash keeps a record next to each file, without the file it would
            # show up in the file manager as an entry that cannot be restored
            trash_info_file(op.dst).unlink(missing_ok=True)
        self.log(f"UNDO {op.kind} {op.dst.name} -> {op.src}")
        return op.src


def move_to_system_trash(path: Path) -> Path:
    """Move the file into the trash of the desktop and return where it is now. Raises if the
    filesystem of the file has no trash, instead of deleting the file for good."""
    file = QFile(path.as_posix())
    if not file.moveToTrash():
        raise OSError(
            f"Cannot move {path} to the system trash ({file.errorString()}). "
            f"Give a trash dir with --trash_dir for files on this filesystem."
        )
    return Path(file.fileName())


def trash_info_file(trashed_file: Path) -> Path:
    """The record the freedesktop trash keeps for a trashed file: <trash>/info/<name>.trashinfo
    for <trash>/files/<name>."""
    return trashed_file.parent.parent / "info" / f"{trashed_file.name}.trashinfo"


def free_path(dst: Path) -> Path:
    """The path itself, or one with a timestamp in the name if something is there already."""
    if not dst.exists():
        return dst
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dst.with_name(f"{dst.stem}.{stamp}{dst.suffix}")


def tags_of(name: str) -> list[str]:
    """The tags of a file name, the whitespace separated words behind the tag marker."""
    stem = Path(name).stem
    if TAG_MARKER not in stem:
        return []
    return stem.split(TAG_MARKER, 1)[1].split()


def name_with_tags(name: str, tags: list[str]) -> str:
    """The file name carrying exactly these tags. Without tags the marker goes away too, so
    tagging and untagging again gives back the original name."""
    stem, suffix = Path(name).stem, Path(name).suffix
    base = stem.split(TAG_MARKER, 1)[0]
    if not tags:
        return base + suffix
    return f"{base}{TAG_MARKER}{' '.join(tags)}{suffix}"


def propose_rename(name: str) -> tuple[str, int]:
    """
    Text and cursor position for the rename dialog. The tag marker is appended to the stem if
    missing, and the cursor goes to the end of the stem, right before the suffix, so typing
    adds tags.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    if TAG_MARKER not in stem:
        stem += TAG_MARKER
    return stem + suffix, len(stem)


def should_rename(old_name: str, new_name: str) -> bool:
    """No rename for an unchanged name, an empty tag list, or something that isn't a file name."""
    if new_name == old_name:
        return False
    if Path(new_name).stem.endswith(TAG_MARKER):
        return False
    if "/" in new_name or new_name.strip() in ("", ".", ".."):
        return False
    return True
