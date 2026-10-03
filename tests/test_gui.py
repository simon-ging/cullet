"""
The viewer driven like a user would: a real window on the offscreen platform, real files in a
temp folder, key presses and actions.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from cullet import cli
from cullet.dedup.image_groups import DedupImageResult, DuplicateGroup, DuplicateMember
from cullet.dedup_review import DedupReview
from cullet.main_window import MainWindow

N_IMAGES = 5


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def folder(tmp_path, monkeypatch):
    # thumbnails are stored in the cache dir, keep them out of the real one
    monkeypatch.setenv("XDG_CACHE_HOME", (tmp_path / "cache").as_posix())
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(N_IMAGES):
        arr = rng.integers(0, 255, (6, 8, 3), dtype=np.uint8).repeat(64, 0).repeat(64, 1)
        Image.fromarray(arr).save(folder / f"img{i}.jpg", quality=92)
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(folder / "sub" / "deep.png")
    return folder


def wait_for(app, condition, what: str) -> None:
    # sleeping instead of QTest.qWait, which keeps the decode threads from getting the GIL
    start = time.time()
    while not condition():
        assert time.time() - start < 10, f"Timeout waiting for {what}"
        time.sleep(0.01)
        app.processEvents()


def wait_for_image(app, window: MainWindow) -> None:
    wait_for(app, lambda: window.loader.get(window.current_path()) is not None, "decode")
    app.processEvents()


def open_window(app, folder: Path, **kwargs) -> MainWindow:
    window = MainWindow(folder, trash_dir=folder.parent / "trash", **kwargs)
    window.resize(1000, 700)
    window.show()
    wait_for_image(app, window)
    return window


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def test_navigation_and_view(app, folder):
    window = open_window(app, folder, start_file=folder / "img2.jpg")
    assert window.current_path().name == "img2.jpg"
    assert len(window.file_model.files) == N_IMAGES
    assert window.view.image_size() == (512, 384)

    QTest.keyClick(window, Qt.Key.Key_Space)
    assert window.current_path().name == "img3.jpg"
    QTest.keyClick(window, Qt.Key.Key_Backspace)
    QTest.keyClick(window, Qt.Key.Key_Home)
    assert window.index == 0
    QTest.keyClick(window, Qt.Key.Key_End)
    assert window.index == N_IMAGES - 1
    window.actions.next_image()
    assert window.index == N_IMAGES - 1, "stepping past the end stays on the last image"
    wait_for_image(app, window)

    window.actions.zoom_actual()
    assert window.view.zoom() == 1.0 and not window.view.fit_mode
    window.actions.zoom_in()
    assert window.view.zoom() > 1.0
    window.actions.toggle_zoom_filter()
    assert not window.view.sharp_zoom
    window.actions.arrow_right()
    window.actions.arrow_down()
    window.actions.arrow_left()
    window.actions.arrow_up()
    window.actions.zoom_out()
    window.actions.zoom_fit()
    assert window.view.fit_mode

    window.actions.toggle_grid()
    assert window.is_grid_mode()
    window.actions.first_image()
    window.actions.arrow_right()
    assert window.index == 1
    window.actions.nav_down()
    window.actions.nav_up()
    window.actions.arrow_down()
    window.actions.arrow_up()
    window.actions.arrow_left()
    assert window.index == 0
    window.actions.toggle_grid()
    assert not window.is_grid_mode()
    window.actions.nav_down()
    assert window.index == 1

    for toggle in (
        window.actions.toggle_thumbnails,
        window.actions.toggle_left_panel,
        window.actions.toggle_bottom,
        window.actions.toggle_info,
        window.actions.toggle_help,
        window.actions.toggle_fullscreen,
    ):
        toggle()
        app.processEvents()
        toggle()
    window.actions.exit_fullscreen()
    assert not window.isFullScreen()

    window.actions.toggle_recursive()
    assert len(window.file_model.files) == N_IMAGES + 1
    window.actions.toggle_recursive()
    window.actions.cycle_sort_order()
    assert len(window.file_model.files) == N_IMAGES

    window.actions.toggle_slideshow()
    assert window.slideshow_timer.isActive()
    seconds = window.slideshow_seconds()
    window.actions.slideshow_slower()
    assert window.slideshow_seconds() > seconds
    window.actions.slideshow_faster()
    window.actions.toggle_slideshow_random()
    window._slideshow_step()
    window.actions.toggle_slideshow()
    assert not window.slideshow_timer.isActive()
    window.actions.quit()


def test_file_operations_and_undo(app, folder):
    good = folder.parent / "good"
    window = open_window(app, folder, target_dirs=[good], tags=["sun"])
    before = names(folder)

    # key 1 is the target folder, key 2 the tag behind it
    QTest.keyClick(window, Qt.Key.Key_2)
    assert (folder / "img0--tags--sun.jpg").is_file()
    assert window.current_path().name == "img0--tags--sun.jpg"
    QTest.keyClick(window, Qt.Key.Key_2)
    assert names(folder) == before

    QTest.keyClick(window, Qt.Key.Key_1)
    assert (good / "img0.jpg").is_file() and not (folder / "img0.jpg").exists()
    assert window.current_path().name == "img1.jpg"

    window.actions.delete_current()
    assert not (folder / "img1.jpg").exists()
    assert len(window.file_model.files) == N_IMAGES - 2

    wait_for_image(app, window)
    window.actions.rotate_right()
    wait_for_image(app, window)
    assert window.view.image_size() == (384, 512)

    for _ in range(5):
        wait_for_image(app, window)
        window.actions.undo()
    assert names(folder) == before
    with Image.open(folder / "img2.jpg") as img:
        assert img.size == (512, 384)
    window.actions.undo()  # nothing left to undo

    # a decode that finishes after its file was rewritten must not reach the view
    window.actions.rotate_right()
    window._on_image_ready(window.current_path().as_posix())
    assert window.view.has_image()
    wait_for_image(app, window)
    assert window.view.image_size() == (384, 512)
    window.close()


def test_review(app, folder):
    def member(name: str, sim: float) -> DuplicateMember:
        path = folder / name
        return DuplicateMember(path, name, 512, 384, path.stat().st_size, sim)

    groups = [
        DuplicateGroup(
            [member("img0.jpg", 0.95), member("img1.jpg", 0.95), member("img2.jpg", 0.9)]
        ),
        DuplicateGroup([member("img3.jpg", 0.99), member("img4.jpg", 0.99)]),
    ]
    review = DedupReview(DedupImageResult(groups, N_IMAGES, 3, {}))
    window = open_window(app, folder, review=review)
    assert [p.name for p in window.file_model.files] == ["img0.jpg", "img1.jpg", "img2.jpg"]

    QTest.keyClick(window, Qt.Key.Key_3)
    assert not (folder / "img2.jpg").exists()
    QTest.keyClick(window, Qt.Key.Key_9)  # no such member, only logged
    QTest.keyClick(window, Qt.Key.Key_J)
    assert review.group_index == 1
    QTest.keyClick(window, Qt.Key.Key_P)
    assert not (folder / "img4.jpg").exists() and (folder / "img3.jpg").is_file()
    assert review.n_unresolved() == 1
    QTest.keyClick(window, Qt.Key.Key_K)
    assert review.group_index == 0

    window.actions.undo()
    assert (folder / "img4.jpg").is_file()
    assert review.group_index == 1, "undo shows the group of the restored file"
    window.close()


@pytest.mark.parametrize(
    "argv, message",
    [
        (["-a", "x", "--dedup"], "use one at a time"),
        (["-T", "two words"], "must be one word"),
        (["-T", "a", "-T", "b"] + ["-a", "x"] * 8, "do not fit"),
    ],
)
def test_cli_rejects_bad_arguments(folder, monkeypatch, argv, message):
    monkeypatch.setattr(sys, "argv", ["cullet", folder.as_posix(), *argv])
    with pytest.raises(ValueError, match=message):
        cli.main()


def test_cli_rejects_missing_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cullet", (tmp_path / "nothing").as_posix()])
    with pytest.raises(FileNotFoundError):
        cli.main()
