import logging
import random
from functools import partial
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QModelIndex, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QLabel,
    QListView,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from cullet.actions import Actions
from cullet.dedup_review import DedupReview, GroupListModel
from cullet.file_ops import FileOps, should_rename, tags_of
from cullet.folder_model import (
    SORT_ORDERS,
    FileListModel,
    list_images,
    make_folder_tree_model,
)
from cullet.image_cache import PRIORITY_WARMUP, ImageLoader, make_thumbnail_decoder
from cullet.image_io import Transform, read_exif_summary
from cullet.image_view import ImageView
from cullet.keymap import (
    DEDUP_KEYMAP,
    DEFAULT_KEYMAP,
    VIEW_KEYS,
    format_help,
    make_tag_keymap,
    make_target_keymap,
    split_action,
)
from cullet.log_panel import LogPanel
from cullet.overlays import Overlays
from cullet.rename_dialog import RenameDialog
from cullet.thumb_store import ThumbnailStore
from cullet.thumbnails import (
    THUMB_DECODE_SIZE,
    ThumbnailGrid,
    ThumbnailModel,
    ThumbnailStrip,
)

PREFETCH_NEIGHBOURS = 2
THUMB_CACHE_ITEMS = 1000
SLIDESHOW_INTERVAL_MS = 3000
SLIDESHOW_MIN_S = 0.2
SLIDESHOW_MAX_S = 60.0
FOLDER_REFRESH_DELAY_MS = 500  # collects bursts of file system events into one refresh
logger = logging.getLogger(__name__)


class _ViewStack(QStackedWidget):
    """Holds the single image view and the grid, tells the overlays when it is resized."""

    resized = Signal()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.resized.emit()


class MainWindow(QMainWindow):
    """
    Left: folder tree, file list, action log. Right: the image or the thumbnail grid, with the
    thumbnail strip below. Status bar at the bottom.

    With a review, the window shows one duplicate group at a time as if it were a folder, and
    the folder tree is replaced by the list of groups.
    """

    def __init__(
        self,
        folder: Path,
        start_file: Path | None = None,
        keymap: dict[str, str] | None = None,
        trash_dir: Path | None = None,
        review: DedupReview | None = None,
        target_dirs: list[Path] | None = None,
        tags: list[str] | None = None,
    ):
        super().__init__()
        self.target_dirs = [Path(target) for target in target_dirs or []]
        self.tags = list(tags or [])
        assert review is None or not (self.target_dirs or self.tags), (
            "Target folders, tags and a duplicate review all want the number keys, "
            "run them one after the other"
        )
        self.folder = folder
        self.listing_title = folder.as_posix()  # shown in the info overlay and window title
        self.review = review
        self.index = -1
        self.recursive = False
        self.sort = SORT_ORDERS[0]
        self.slideshow_random = False
        self.left_wanted = True
        self.bottom_wanted = True
        self.strip_wanted = True
        self._exif_line = ""
        self.thumb_store: ThumbnailStore | None = None

        self.log_panel = LogPanel()
        self.file_ops = FileOps(trash_dir, self.log_panel.log)

        self.loader = ImageLoader(parent=self)
        self.loader.image_ready.connect(self._on_image_ready)
        self.loader.image_failed.connect(self._on_image_failed)

        self.tree_model = make_folder_tree_model(self)
        self.tree = QTreeView()
        self.tree.setModel(self.tree_model)
        self.tree.setRootIndex(self.tree_model.index(self.tree_model.rootPath()))
        for column in range(1, self.tree_model.columnCount()):
            self.tree.hideColumn(column)
        self.tree.setHeaderHidden(True)
        self.tree.clicked.connect(self._on_tree_clicked)

        self.group_list = QListView()
        if review is not None:
            self.group_model = GroupListModel(review, self)
            self.group_list.setModel(self.group_model)
            self.group_list.clicked.connect(lambda index: self.go_to_group(index.row()))
        self.group_list.setVisible(review is not None)
        self.tree.setVisible(review is None)

        self.file_model = FileListModel(self)
        self.file_list = QListView()
        self.file_list.setModel(self.file_model)
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_list.selectionModel().currentRowChanged.connect(self._on_current_row_changed)
        self.file_list.clicked.connect(lambda _index: self.view.setFocus())

        self.view = ImageView()
        self.view.zoom_changed.connect(lambda _zoom: self._update_status())
        self.view.wheel_navigate.connect(lambda steps: self.go_to(self.index + steps))
        self.view.double_clicked.connect(lambda: self.set_fullscreen(not self.isFullScreen()))

        self.thumb_loader = ImageLoader(
            make_thumbnail_decoder(THUMB_DECODE_SIZE),
            max_items=THUMB_CACHE_ITEMS,
            max_bytes=None,
            parent=self,
        )
        self.thumb_model = ThumbnailModel(self.thumb_loader, self)
        self.strip = ThumbnailStrip()
        self.strip.setModel(self.thumb_model)
        self.strip.selectionModel().currentRowChanged.connect(self._on_current_row_changed)
        self.strip.wheel_navigate.connect(lambda steps: self.go_to(self.index + steps))
        self.grid = ThumbnailGrid()
        self.grid.setModel(self.thumb_model)
        self.grid.selectionModel().currentRowChanged.connect(self._on_current_row_changed)
        self.grid.wheel_navigate.connect(lambda steps: self.go_to(self.index + steps))
        self.grid.doubleClicked.connect(lambda _index: self.set_grid_mode(False))

        self.stack = _ViewStack()
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.grid)
        self.overlays = Overlays(self.stack)
        self.stack.resized.connect(self.overlays.relayout)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.stack, 1)
        right_layout.addWidget(self.strip)

        self.left_panel = QSplitter(Qt.Orientation.Vertical)
        self.left_panel.addWidget(self.tree)
        self.left_panel.addWidget(self.group_list)
        self.left_panel.addWidget(self.file_list)
        self.left_panel.addWidget(self.log_panel)
        self.left_panel.setSizes([400, 400, 400, 200])
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1100])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.status_left = QLabel()
        self.status_right = QLabel()
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addPermanentWidget(self.status_right)

        self.slideshow_timer = QTimer(self)
        self.slideshow_timer.setInterval(SLIDESHOW_INTERVAL_MS)
        self.slideshow_timer.timeout.connect(self._slideshow_step)

        # pick up files added or removed by other programs
        self.watcher = QFileSystemWatcher(self)
        self.watcher.directoryChanged.connect(lambda _path: self._refresh_timer.start())
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(FOLDER_REFRESH_DELAY_MS)
        self._refresh_timer.timeout.connect(self._on_folder_changed_on_disk)

        self.actions = Actions(self)
        if keymap is None:
            keymap = DEFAULT_KEYMAP if review is None else DEDUP_KEYMAP
            if self.target_dirs:
                keymap = {**keymap, **make_target_keymap(len(self.target_dirs))}
            if self.tags:
                keymap = {**keymap, **make_tag_keymap(len(self.tags), len(self.target_dirs))}
        self._bind_keys(keymap)
        self.overlays.set_help_text(format_help(keymap, self.actions))

        for number, target in enumerate(self.target_dirs, start=1):
            self.log_panel.log(f"TARGET {number} = {target}")
        for number, tag in enumerate(self.tags, start=len(self.target_dirs) + 1):
            self.log_panel.log(f"TAG {number} = {tag}")

        self.set_folder(folder)
        if review is not None:
            self.log_panel.log(review.summary())
            self.go_to_group(0)
        elif start_file is not None:
            self.go_to(self.file_model.files.index(start_file))
        self.view.setFocus()

    # ---------- folder and navigation

    def set_folder(self, folder: Path) -> None:
        self.folder = folder
        self.thumb_loader.cancel_queued()
        self.thumb_store = ThumbnailStore(folder, THUMB_DECODE_SIZE)
        self.thumb_loader.set_decode_fn(make_thumbnail_decoder(THUMB_DECODE_SIZE, self.thumb_store))
        tree_index = self.tree_model.index(folder.as_posix())
        self.tree.setCurrentIndex(tree_index)
        self.tree.expand(tree_index)
        self.tree.scrollTo(tree_index)
        self.refresh_folder()
        logger.info(f"Folder {folder} with {len(self.file_model.files)} images")

    def refresh_folder(self, show: Path | None = None) -> None:
        """Re-read the folder, or in a review the current group. Shows the given file, else
        stays at the same position."""
        if self.review is not None:
            self._show_group(show, advance=True)
            return
        self.show_files(
            list_images(self.folder, self.recursive, self.sort),
            self.folder.as_posix(),
            show=show,
        )

    def show_files(
        self,
        files: list[Path],
        title: str,
        labels: list[str] | None = None,
        show: Path | None = None,
    ) -> None:
        """List these files in the file list, strip and grid. Labels replace the relative paths
        in the file list."""
        old_index = self.index
        self.file_model.set_files(files, self.folder, labels)
        self.thumb_model.set_files(files)
        self.index = -1
        self.listing_title = title
        self.setWindowTitle(f"{self.folder.name} - cullet")
        self._watch_folders(files)
        # fill the thumbnail store for the whole folder while the user looks at the first images
        for path in files:
            self.thumb_loader.request(path, PRIORITY_WARMUP)
        if len(files) == 0:
            self.view.clear()
            self._update_status()
            return
        if show is not None and show in files:
            self.go_to(files.index(show))
        else:
            self.go_to(max(old_index, 0))

    def go_to(self, index: int) -> None:
        files = self.file_model.files
        if len(files) == 0:
            return
        self.index = min(max(index, 0), len(files) - 1)
        self._exif_line = read_exif_summary(files[self.index])
        self.file_list.setCurrentIndex(self.file_model.index(self.index))
        thumb_index = self.thumb_model.index(self.index)
        for thumb_view in (self.strip, self.grid):
            thumb_view.setCurrentIndex(thumb_index)
            thumb_view.scroll_to_row(thumb_index)
        self._show_current()
        for offset in range(1, PREFETCH_NEIGHBOURS + 1):
            for neighbour in (self.index + offset, self.index - offset):
                if 0 <= neighbour < len(files):
                    self.loader.request(files[neighbour])

    def current_path(self) -> Path | None:
        if self.index < 0:
            return None
        return self.file_model.files[self.index]

    def set_recursive(self, recursive: bool) -> None:
        self.recursive = recursive
        self.refresh_folder(show=self.current_path())

    def cycle_sort_order(self) -> None:
        self.sort = SORT_ORDERS[(SORT_ORDERS.index(self.sort) + 1) % len(SORT_ORDERS)]
        self.log_panel.log(f"SORT by {self.sort}")
        self.refresh_folder(show=self.current_path())

    def _watch_folders(self, files: list[Path]) -> None:
        watched = self.watcher.directories()
        if watched:
            self.watcher.removePaths(watched)
        folders = {self.folder.as_posix()} | {p.parent.as_posix() for p in files}
        self.watcher.addPaths(sorted(folders))

    def _on_folder_changed_on_disk(self) -> None:
        # a review lists members of groups, not a folder, and is the only thing deleting them
        if self.review is not None:
            return
        # the viewer's own operations refresh right away, so only react to real differences
        if list_images(self.folder, self.recursive, self.sort) == self.file_model.files:
            return
        self.log_panel.log(f"REFRESH {self.folder.name} changed on disk")
        self.refresh_folder(show=self.current_path())

    # ---------- duplicate review

    def go_to_group(self, group_index: int, show: Path | None = None) -> None:
        review = self.review
        review.group_index = min(max(group_index, 0), len(review.groups) - 1)
        self._show_group(show)

    def _show_group(self, show: Path | None = None, advance: bool = False) -> None:
        """List the current group. After undo the restored file's group is shown instead. With
        advance, a group that is down to one file moves on to the next unresolved one."""
        review = self.review
        if show is not None and review.group_of(show) is not None:
            review.group_index = review.group_of(show)
        elif advance and review.is_resolved(review.group_index):
            next_index = review.next_unresolved(review.group_index, 1)
            if next_index is None:
                next_index = review.next_unresolved(review.group_index, -1)
            if next_index is None:
                self.log_panel.log("DEDUP all groups resolved")
            else:
                review.group_index = next_index
        group_index = review.group_index
        self.group_model.refresh()
        self.group_list.setCurrentIndex(self.group_model.index(group_index))
        self.group_list.scrollTo(self.group_model.index(group_index))
        title = f"group {group_index + 1}/{len(review.groups)}, {review.n_unresolved()} open"
        self.show_files(review.files(group_index), title, review.labels(group_index), show=show)

    def delete_member(self, number: int) -> None:
        """Trash image number (1-based) of the current group."""
        files = self.file_model.files
        if not 1 <= number <= len(files):
            self.log_panel.log(f"DEL no image {number}, the group has {len(files)}")
            return
        path = files[number - 1]
        if self._trash(path, note=self.review.member_note(path)):
            self.refresh_folder()

    def accept_proposal(self) -> None:
        """Trash every member of the current group except the proposed keep."""
        review = self.review
        keep = review.groups[review.group_index].keep
        if not keep.path.exists():
            self.log_panel.log(f"ACCEPT skipped, proposed keep {keep.path.name} is gone")
            return
        removals = [m for m in review.existing(review.group_index) if m is not keep]
        n_trashed = sum(
            self._trash(member.path, note=review.member_note(member.path)) for member in removals
        )
        self.log_panel.log(f"ACCEPT kept {keep.path.name}, trashed {n_trashed}")
        self.refresh_folder()

    # ---------- file operations

    def move_to_target(self, number: int) -> None:
        """Move the current image into target folder number (1-based)."""
        if not 1 <= number <= len(self.target_dirs):
            self.log_panel.log(f"MV no target {number}, there are {len(self.target_dirs)}")
            return
        path = self.current_path()
        if path is None:
            return
        self.file_ops.move(path, self.target_dirs[number - 1])
        self.loader.invalidate(path)
        self.refresh_folder()

    def toggle_tag(self, number: int) -> None:
        """Add tag number (1-based) to the name of the current image, or remove it again."""
        if not 1 <= number <= len(self.tags):
            self.log_panel.log(f"TAG no tag {number}, there are {len(self.tags)}")
            return
        path = self.current_path()
        if path is None:
            return
        new_path = self.file_ops.toggle_tag(path, self.tags[number - 1])
        self.refresh_folder(show=new_path)

    def rotate_current(self, transform: Transform) -> None:
        path = self.current_path()
        if path is None:
            return
        self.file_ops.rotate(path, transform)
        self.loader.invalidate(path)
        self.thumb_model.refresh(path)
        self._show_current()

    def delete_current(self) -> None:
        path = self.current_path()
        if path is None:
            return
        if self._trash(path):
            self.refresh_folder()

    def _trash(self, path: Path, note: str = "") -> bool:
        """Trash the file. A filesystem without a trash refuses, then the file stays where it
        is and the reason goes to the log panel."""
        try:
            self.file_ops.trash(path, note=note)
        except OSError as e:
            # the viewer keeps running, the user can go on with other files or restart with
            # a trash dir
            self.log_panel.log(f"DEL failed: {e}")
            return False
        self.loader.invalidate(path)
        return True

    def rename_current(self) -> None:
        path = self.current_path()
        if path is None:
            return
        dialog = RenameDialog(path.name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_name = dialog.new_name()
        if not should_rename(path.name, new_name):
            return
        new_path = self.file_ops.rename(path, new_name)
        self.refresh_folder(show=new_path)

    def undo(self) -> None:
        restored = self.file_ops.undo()
        if restored is None:
            return
        self.loader.invalidate(restored)
        self.thumb_loader.invalidate(restored)
        self.refresh_folder(show=restored)

    # ---------- view modes and panels

    def set_slideshow(self, on: bool) -> None:
        if on:
            self.slideshow_timer.start()
            self.log_panel.log(f"SLIDESHOW on, {self.slideshow_seconds():.1f} s per image")
        else:
            self.slideshow_timer.stop()
            self.log_panel.log("SLIDESHOW off")
        self._update_status()

    def slideshow_seconds(self) -> float:
        return self.slideshow_timer.interval() / 1000

    def change_slideshow_speed(self, factor: float) -> None:
        seconds = min(SLIDESHOW_MAX_S, max(SLIDESHOW_MIN_S, self.slideshow_seconds() * factor))
        self.slideshow_timer.setInterval(round(seconds * 1000))
        self.log_panel.log(f"SLIDESHOW {seconds:.1f} s per image")
        self._update_status()

    def set_slideshow_random(self, on: bool) -> None:
        self.slideshow_random = on
        self.log_panel.log(f"SLIDESHOW random order {'on' if on else 'off'}")
        self._update_status()

    def set_sharp_zoom(self, sharp: bool) -> None:
        self.view.set_sharp_zoom(sharp)
        self.log_panel.log(f"ZOOM pixels {'sharp' if sharp else 'smooth'} above 1:1")
        self._update_status()

    def _slideshow_step(self) -> None:
        n_files = len(self.file_model.files)
        if n_files == 0:
            return
        if self.slideshow_random and n_files > 1:
            candidates = [i for i in range(n_files) if i != self.index]
            self.go_to(random.choice(candidates))
            return
        self.go_to((self.index + 1) % n_files)

    def set_fullscreen(self, fullscreen: bool) -> None:
        if fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

    def is_grid_mode(self) -> bool:
        return self.stack.currentWidget() is self.grid

    def set_grid_mode(self, grid: bool) -> None:
        self.stack.setCurrentWidget(self.grid if grid else self.view)
        self.apply_panels()
        if grid:
            self.grid.setFocus()
            if self.index >= 0:
                self.grid.scroll_to_row(self.thumb_model.index(self.index))
        else:
            self.view.setFocus()
        self.overlays.relayout()

    def apply_panels(self) -> None:
        self.left_panel.setVisible(self.left_wanted)
        self.statusBar().setVisible(self.bottom_wanted)
        # the strip repeats the grid, so it is hidden while the grid is shown
        self.strip.setVisible(self.strip_wanted and self.bottom_wanted and not self.is_grid_mode())

    # ---------- internals

    def _show_current(self) -> None:
        path = self.current_path()
        qimage = self.loader.get(path)
        if qimage is not None:
            self.view.set_image(qimage)
        self._update_status()

    def _on_image_ready(self, key: str) -> None:
        path = self.current_path()
        if path is None or path.as_posix() != key:
            return
        qimage = self.loader.get(path)
        if qimage is None:
            # the file was rewritten while this decode ran, e.g. rotated. The lookup just
            # started a decode of the new content, which reports back here when it is done.
            return
        self.view.set_image(qimage)
        self._update_status()

    def _on_image_failed(self, key: str, message: str) -> None:
        self.status_left.setText(f"{key}: {message}")

    def _on_current_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if current.isValid() and current.row() != self.index:
            self.go_to(current.row())

    def _on_tree_clicked(self, index: QModelIndex) -> None:
        folder = Path(self.tree_model.filePath(index))
        if folder != self.folder:
            self.set_folder(folder)
        self.view.setFocus()

    def _update_status(self) -> None:
        path = self.current_path()
        if path is None:
            self.status_left.setText(f"{self.listing_title}: no images")
            self.status_right.setText("")
            self.overlays.set_info(f"{self.listing_title}  0/0", "")
            return
        self.status_left.setText(path.as_posix())
        if self.view.has_image():
            width, height = self.view.image_size()
            resolution = f"{width}x{height}"
            dims = f"{resolution}  {self.view.zoom() * 100:.0f}%"
        else:
            resolution = ""
            dims = "loading"
        position = f"{self.index + 1}/{len(self.file_model.files)}"
        slideshow = ""
        if self.slideshow_timer.isActive():
            random_flag = " random" if self.slideshow_random else ""
            slideshow = f"slideshow {self.slideshow_seconds():.1f}s{random_flag}  "
        recursive = "recursive  " if self.recursive else ""
        sort = f"sort {self.sort}  " if self.sort != SORT_ORDERS[0] else ""
        smooth = "" if self.view.sharp_zoom else "smooth  "
        self.status_right.setText(f"{slideshow}{recursive}{sort}{smooth}{dims}  {position}")
        # in a review the listing is a group of files from anywhere, so the folder of the
        # current image goes on the first line and the group gets one of its own
        if self.review is None:
            left = f"{self.listing_title}  {position}"
        else:
            left = f"{path.parent.as_posix()}  {position}\n{self.listing_title}"
        tags = " ".join(f"#{tag}" for tag in tags_of(path.name))
        # the resolution leads the exif line, it is the part that is looked up most
        top_right = "  ".join(part for part in (resolution, self._exif_line) if part)
        self.overlays.set_info(left, path.name, top_right, tags)

    def _bind_keys(self, keymap: dict[str, str]) -> None:
        for key, value in keymap.items():
            action_name, args = split_action(value)
            # getattr fails loudly for a typo in the keymap
            action = partial(getattr(self.actions, action_name), *args)
            if key in VIEW_KEYS:
                shortcut = QShortcut(QKeySequence(key), self.stack)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            else:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(action)
