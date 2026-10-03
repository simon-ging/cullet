import shutil

import numpy as np
import pytest
from PIL import Image

from cullet.file_ops import (
    TAG_MARKER,
    FileOps,
    name_with_tags,
    propose_rename,
    should_rename,
    tags_of,
)
from cullet.image_io import (
    EXIF_DATETIME_ORIGINAL_TAG,
    EXIF_DATETIME_TAG,
    EXIF_EXPOSURE_TIME_TAG,
    EXIF_FNUMBER_TAG,
    EXIF_IFD_TAG,
    EXIF_ISO_TAG,
    EXIF_MODEL_TAG,
    Transform,
    read_exif_summary,
)
from cullet.thumb_store import ThumbnailStore, reset_thumb_stores


@pytest.mark.parametrize(
    "name,expected_text,expected_cursor",
    [
        ("a.jpg", f"a{TAG_MARKER}.jpg", len(f"a{TAG_MARKER}")),
        (f"a{TAG_MARKER}sun.jpg", f"a{TAG_MARKER}sun.jpg", len(f"a{TAG_MARKER}sun")),
        ("noext", f"noext{TAG_MARKER}", len(f"noext{TAG_MARKER}")),
        (
            "IMG_2026.01.02.jpeg",
            f"IMG_2026.01.02{TAG_MARKER}.jpeg",
            len(f"IMG_2026.01.02{TAG_MARKER}"),
        ),
    ],
)
def test_propose_rename(name, expected_text, expected_cursor):
    assert propose_rename(name) == (expected_text, expected_cursor)


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("a.jpg", "a.jpg", False),
        ("a.jpg", f"a{TAG_MARKER}.jpg", False),
        ("a.jpg", f"a{TAG_MARKER}sun.jpg", True),
        (f"a{TAG_MARKER}sun.jpg", f"a{TAG_MARKER}sun beach.jpg", True),
        ("a.jpg", "b.jpg", True),
        ("a.jpg", "sub/b.jpg", False),
        ("a.jpg", "", False),
        ("a.jpg", "..", False),
    ],
)
def test_should_rename(old, new, expected):
    assert should_rename(old, new) is expected


def make_jpeg(path, width=48, height=32):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, : width // 2] = (255, 0, 0)
    Image.fromarray(arr).save(path, quality=95)


def test_file_ops_trash_rename_rotate_undo(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    trash = tmp_path / "trash"
    src = folder / "a.jpg"
    make_jpeg(src)
    messages = []
    ops = FileOps(trash, messages.append)

    dst = ops.trash(src)
    assert not src.exists() and dst.exists()
    assert dst == trash / src.relative_to(src.anchor)
    assert messages[-1].startswith("DEL a.jpg")

    assert ops.undo() == src
    assert src.exists() and not dst.exists()
    assert messages[-1].startswith("UNDO DEL")

    renamed = ops.rename(src, f"a{TAG_MARKER}sun.jpg")
    assert renamed.exists() and not src.exists()
    assert messages[-1] == f"MV a.jpg -> a{TAG_MARKER}sun.jpg"
    assert ops.undo() == src
    assert src.exists()

    if shutil.which("jpegtran") is None:
        pytest.skip("jpegtran not installed")
    ops.rotate(src, Transform.ROTATE_90)
    with Image.open(src) as img:
        assert img.size == (32, 48)
    assert ops.undo() == src
    with Image.open(src) as img:
        assert img.size == (48, 32)
    assert ops.undo() is None
    assert messages[-1] == "UNDO nothing to undo"


def test_trash_twice_keeps_both(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    ops = FileOps(tmp_path / "trash", lambda _m: None)
    src = folder / "a.jpg"
    make_jpeg(src)
    first = ops.trash(src)
    make_jpeg(src)
    second = ops.trash(src)
    assert first != second and first.exists() and second.exists()


def test_move_to_target_and_undo(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    target = tmp_path / "good"
    messages = []
    ops = FileOps(tmp_path / "trash", messages.append)
    src = folder / "a.jpg"
    make_jpeg(src)

    dst = ops.move(src, target)
    assert dst == target / "a.jpg"
    assert dst.exists() and not src.exists(), "the target dir is created on the first move"
    assert messages[-1] == f"MV a.jpg -> {dst}"

    make_jpeg(src)
    second = ops.move(src, target)
    assert second != dst and dst.exists() and second.exists()

    assert ops.undo() == src
    assert src.exists() and not second.exists()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.jpg", []),
        (f"a{TAG_MARKER}.jpg", []),
        (f"a{TAG_MARKER}sun.jpg", ["sun"]),
        (f"a{TAG_MARKER}sun beach.jpg", ["sun", "beach"]),
        (f"IMG_2026.01.02{TAG_MARKER}sun.jpeg", ["sun"]),
    ],
)
def test_tags_of(name, expected):
    assert tags_of(name) == expected
    assert name_with_tags(name, expected) in (name, name.replace(TAG_MARKER, ""))


def test_toggle_tag(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    messages = []
    ops = FileOps(tmp_path / "trash", messages.append)
    path = folder / "a.jpg"
    make_jpeg(path)

    tagged = ops.toggle_tag(path, "sun")
    assert tagged.name == f"a{TAG_MARKER}sun.jpg" and tagged.exists()
    two = ops.toggle_tag(tagged, "beach")
    assert two.name == f"a{TAG_MARKER}sun beach.jpg"
    assert tags_of(ops.toggle_tag(two, "sun").name) == ["beach"], "a second press removes the tag"

    back = ops.toggle_tag(folder / f"a{TAG_MARKER}beach.jpg", "beach")
    assert back == path, "without tags the marker goes away again"
    assert ops.undo() and ops.undo() and ops.undo() and ops.undo() == path
    assert (folder / "a.jpg").exists(), "every toggle is one undo step"


def test_thumb_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cullet.thumb_store.get_cache_dir", lambda: tmp_path / "cache")
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)
    store = ThumbnailStore(folder, 256)
    path = folder / "sub" / "a.jpg"
    assert store.get(path, 1.0) is None
    store.put(path, 1.0, b"thumb")
    assert store.get(path, 1.0) == b"thumb"
    assert store.get(path, 2.0) is None, "changed mtime must miss"
    assert ThumbnailStore(folder, 128).get(path, 1.0) is None, "other size must miss"
    assert ThumbnailStore(folder, 256).get(path, 1.0) == b"thumb", "must persist on disk"
    assert store.key(path) == "sub/a.jpg"


def test_reset_thumb_stores(tmp_path, monkeypatch):
    monkeypatch.setattr("cullet.thumb_store.get_cache_dir", lambda: tmp_path / "cache")
    folder = tmp_path / "photos"
    sub = folder / "sub"
    sibling = tmp_path / "photos_other"
    for target in (folder, sub, sibling):
        target.mkdir(parents=True)
        ThumbnailStore(target, 256).put(target / "a.jpg", 1.0, b"thumb")

    deleted = reset_thumb_stores(folder)
    assert len(deleted) >= 2, deleted
    assert ThumbnailStore(folder, 256).get(folder / "a.jpg", 1.0) is None
    assert ThumbnailStore(sub, 256).get(sub / "a.jpg", 1.0) is None, "subfolders go too"
    other = ThumbnailStore(sibling, 256)
    assert other.get(sibling / "a.jpg", 1.0) == b"thumb", "a folder with the same prefix stays"


def test_read_exif_summary(tmp_path):
    path = tmp_path / "a.jpg"
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[EXIF_MODEL_TAG] = "Pixel 9"
    exif[EXIF_DATETIME_TAG] = "2026:01:02 03:04:05"
    ifd = exif.get_ifd(EXIF_IFD_TAG)
    ifd[EXIF_DATETIME_ORIGINAL_TAG] = "2025:12:31 23:59:58"
    ifd[EXIF_EXPOSURE_TIME_TAG] = 0.004
    ifd[EXIF_FNUMBER_TAG] = 1.8
    ifd[EXIF_ISO_TAG] = 200
    Image.fromarray(arr).save(path, exif=exif.tobytes())
    summary = read_exif_summary(path)
    assert summary.startswith(
        "2025:12:31 23:59:58  Pixel 9  1/250s  f/1.8  ISO 200  0.0 MB"
    ), summary


def test_read_exif_summary_without_exif(tmp_path):
    path = tmp_path / "a.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    summary = read_exif_summary(path)
    assert summary.startswith("mtime 20") and summary.endswith("  0.0 MB"), summary
