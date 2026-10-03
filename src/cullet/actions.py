"""
Every user-facing operation of the viewer. Key bindings map to method names of this class,
and scripts can call the methods directly. The docstrings are the help overlay, keep them short.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cullet.image_io import Transform
from cullet.image_view import WHEEL_ZOOM_STEP

SLIDESHOW_SPEED_FACTOR = 1.5

if TYPE_CHECKING:
    from cullet.main_window import MainWindow


class Actions:
    def __init__(self, window: MainWindow):
        self.window = window

    # ---------- navigation

    def next_image(self) -> None:
        """next image"""
        self.window.go_to(self.window.index + 1)

    def prev_image(self) -> None:
        """previous image"""
        self.window.go_to(self.window.index - 1)

    def first_image(self) -> None:
        """first image"""
        self.window.go_to(0)

    def last_image(self) -> None:
        """last image"""
        self.window.go_to(len(self.window.file_model.files) - 1)

    # in the single view the arrow keys never reach these shortcuts: ImageView claims them and
    # pans continuously while a key is held. The pan branch stays for scripting.

    def arrow_left(self) -> None:
        """pan image, in grid: previous image"""
        if self.window.is_grid_mode():
            self.prev_image()
            return
        self.window.view.pan(-self.window.view.pan_step()[0], 0)

    def arrow_right(self) -> None:
        """pan image, in grid: next image"""
        if self.window.is_grid_mode():
            self.next_image()
            return
        self.window.view.pan(self.window.view.pan_step()[0], 0)

    def arrow_up(self) -> None:
        """pan image, in grid: one row up"""
        if self.window.is_grid_mode():
            self.nav_up()
            return
        self.window.view.pan(0, -self.window.view.pan_step()[1])

    def arrow_down(self) -> None:
        """pan image, in grid: one row down"""
        if self.window.is_grid_mode():
            self.nav_down()
            return
        self.window.view.pan(0, self.window.view.pan_step()[1])

    # w/a/s/d never pan, they navigate in both views, so they work without looking at the mode

    def nav_up(self) -> None:
        """grid: one row up, else previous image"""
        if self.window.is_grid_mode():
            self.window.go_to(self.window.index - self.window.grid.columns())
            return
        self.prev_image()

    def nav_down(self) -> None:
        """grid: one row down, else next image"""
        if self.window.is_grid_mode():
            self.window.go_to(self.window.index + self.window.grid.columns())
            return
        self.next_image()

    # ---------- duplicate review, only bound in DEDUP_KEYMAP

    def next_group(self) -> None:
        """next duplicate group"""
        self.window.go_to_group(self.window.review.group_index + 1)

    def prev_group(self) -> None:
        """previous duplicate group"""
        self.window.go_to_group(self.window.review.group_index - 1)

    def delete_member(self, number: int) -> None:
        """move image N of the group to the trash"""
        self.window.delete_member(number)

    def accept_proposal(self) -> None:
        """accept: trash all DEL members of the group"""
        self.window.accept_proposal()

    # ---------- file operations

    def move_to_target(self, number: int) -> None:
        """move the image into target folder N"""
        self.window.move_to_target(number)

    def toggle_tag(self, number: int) -> None:
        """add or remove tag N in the file name"""
        self.window.toggle_tag(number)

    def rotate_left(self) -> None:
        """rotate left, saved to disk"""
        self.window.rotate_current(Transform.ROTATE_270)

    def rotate_right(self) -> None:
        """rotate right, saved to disk"""
        self.window.rotate_current(Transform.ROTATE_90)

    def delete_current(self) -> None:
        """move to the trash"""
        self.window.delete_current()

    def rename_current(self) -> None:
        """rename, cursor placed for adding --tags--"""
        self.window.rename_current()

    def undo(self) -> None:
        """undo the last file operation"""
        self.window.undo()

    # ---------- view

    def zoom_in(self) -> None:
        """zoom in"""
        self.window.view.zoom_by(WHEEL_ZOOM_STEP)

    def zoom_out(self) -> None:
        """zoom out"""
        self.window.view.zoom_by(1 / WHEEL_ZOOM_STEP)

    def zoom_fit(self) -> None:
        """fit to window"""
        self.window.view.fit()

    def zoom_actual(self) -> None:
        """zoom 1:1"""
        self.window.view.zoom_actual()

    def toggle_zoom_filter(self) -> None:
        """zoomed in: sharp pixels or smooth"""
        self.window.set_sharp_zoom(not self.window.view.sharp_zoom)

    def toggle_slideshow(self) -> None:
        """slideshow"""
        self.window.set_slideshow(not self.window.slideshow_timer.isActive())

    def slideshow_slower(self) -> None:
        """slideshow slower"""
        self.window.change_slideshow_speed(SLIDESHOW_SPEED_FACTOR)

    def slideshow_faster(self) -> None:
        """slideshow faster"""
        self.window.change_slideshow_speed(1 / SLIDESHOW_SPEED_FACTOR)

    def toggle_slideshow_random(self) -> None:
        """slideshow in random order"""
        self.window.set_slideshow_random(not self.window.slideshow_random)

    def toggle_recursive(self) -> None:
        """include subfolders"""
        self.window.set_recursive(not self.window.recursive)

    def cycle_sort_order(self) -> None:
        """sort order: name, mtime, exif date"""
        self.window.cycle_sort_order()

    def toggle_fullscreen(self) -> None:
        """fullscreen"""
        self.window.set_fullscreen(not self.window.isFullScreen())

    def toggle_grid(self) -> None:
        """thumbnail grid instead of the image"""
        self.window.set_grid_mode(not self.window.is_grid_mode())

    def toggle_thumbnails(self) -> None:
        """thumbnail strip"""
        self.window.strip_wanted = not self.window.strip_wanted
        self.window.apply_panels()

    def toggle_left_panel(self) -> None:
        """left panel: folders, files, log"""
        self.window.left_wanted = not self.window.left_wanted
        self.window.apply_panels()

    def toggle_bottom(self) -> None:
        """bottom: strip and status bar"""
        self.window.bottom_wanted = not self.window.bottom_wanted
        self.window.apply_panels()

    def toggle_info(self) -> None:
        """folder, filename and EXIF overlay"""
        self.window.overlays.toggle_info()

    def toggle_help(self) -> None:
        """this help"""
        self.window.overlays.toggle_help()

    def exit_fullscreen(self) -> None:
        """leave fullscreen"""
        # outside fullscreen this does nothing at all, so no stray key quits the viewer or
        # un-maximizes the window
        if self.window.isFullScreen():
            self.window.set_fullscreen(False)

    def quit(self) -> None:
        """quit"""
        self.window.close()
