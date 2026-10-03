"""
Review state for the duplicate groups found by cullet.dedup: which group is shown, which members
still exist on disk, and the labels the viewer shows for them. Groups never change, a member
that was moved to the trash simply stops being listed, and reappears after undo.
"""

from pathlib import Path

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, Qt

from cullet.dedup.image_groups import DedupImageResult, DuplicateGroup, DuplicateMember, describe


class DedupReview:
    def __init__(self, result: DedupImageResult):
        self.result = result
        self.groups: list[DuplicateGroup] = result.groups
        self.group_index = 0
        self._member_of: dict[str, DuplicateMember] = {
            m.path.as_posix(): m for g in self.groups for m in g.members
        }
        self._group_of: dict[str, int] = {
            m.path.as_posix(): i for i, g in enumerate(self.groups) for m in g.members
        }

    def summary(self) -> str:
        return (
            f"DEDUP {self.result.n_images} images, {self.result.n_pairs} similar pairs forming "
            f"{len(self.groups)} duplicate groups"
        )

    def existing(self, group_index: int) -> list[DuplicateMember]:
        return [m for m in self.groups[group_index].members if m.path.exists()]

    def files(self, group_index: int) -> list[Path]:
        return [m.path for m in self.existing(group_index)]

    def labels(self, group_index: int) -> list[str]:
        """One file list row per existing member: KEEP/DEL, dimensions, size, sim, path."""
        keep = self.groups[group_index].keep
        return [
            f"{'KEEP' if m is keep else 'DEL '} {describe(m)} sim={m.sim:.3f}  {m.rel_file}"
            for m in self.existing(group_index)
        ]

    def is_resolved(self, group_index: int) -> bool:
        return len(self.existing(group_index)) <= 1

    def n_unresolved(self) -> int:
        return sum(not self.is_resolved(i) for i in range(len(self.groups)))

    def next_unresolved(self, start: int, step: int) -> int | None:
        """Nearest unresolved group after start in the given direction, without wrapping."""
        i = start + step
        while 0 <= i < len(self.groups):
            if not self.is_resolved(i):
                return i
            i += step
        return None

    def group_of(self, path: Path) -> int | None:
        return self._group_of.get(Path(path).as_posix())

    def member_note(self, path: Path) -> str:
        """Dimensions, size and similarity of a member, for the action log."""
        member = self._member_of[Path(path).as_posix()]
        return f"{describe(member)} sim={member.sim:.3f}"

    def group_summary(self, group_index: int) -> str:
        group = self.groups[group_index]
        existing = self.existing(group_index)
        state = "done" if len(existing) <= 1 else "    "
        max_sim = max(m.sim for m in group.members)
        return (
            f"{state} {len(existing)}/{len(group.members)} imgs  sim {max_sim:.3f}  "
            f"{group.keep.path.name}"
        )


class GroupListModel(QAbstractListModel):
    """One row per duplicate group, for the panel that replaces the folder tree."""

    def __init__(self, review: DedupReview, parent: QObject | None = None):
        super().__init__(parent)
        self.review = review

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.review.groups)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.review.group_summary(index.row())
        if role == Qt.ItemDataRole.ToolTipRole:
            return "\n".join(m.rel_file for m in self.review.groups[index.row()].members)
        return None

    def refresh(self) -> None:
        """Members were trashed or restored, the rows show new counts."""
        if len(self.review.groups) == 0:
            return
        self.dataChanged.emit(self.index(0), self.index(len(self.review.groups) - 1))
