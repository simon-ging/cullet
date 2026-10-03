import os

import numpy as np
import pytest
from PIL import Image

from cullet.folder_model import SORT_ORDERS, list_images
from cullet.image_io import EXIF_DATETIME_ORIGINAL_TAG, EXIF_IFD_TAG


def save(path, exif_date: str | None = None, mtime: float | None = None):
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    if exif_date is None:
        img.save(path)
    else:
        exif = Image.Exif()
        exif.get_ifd(EXIF_IFD_TAG)[EXIF_DATETIME_ORIGINAL_TAG] = exif_date
        img.save(path, exif=exif.tobytes())
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_list_images_sort_orders(tmp_path):
    # name order: a, b, c. mtime order: c, a, b. exif order: b (2020), c (no exif -> mtime
    # 2022), a (2024)
    save(tmp_path / "a.jpg", exif_date="2024:01:01 00:00:00", mtime=2_000_000)
    save(tmp_path / "b.jpg", exif_date="2020:01:01 00:00:00", mtime=3_000_000)
    save(tmp_path / "c.jpg", mtime=1_000_000)
    (tmp_path / "notes.txt").write_text("ignored")
    names = lambda files: [p.name for p in files]
    assert names(list_images(tmp_path, sort="name")) == ["a.jpg", "b.jpg", "c.jpg"]
    assert names(list_images(tmp_path, sort="mtime")) == ["c.jpg", "a.jpg", "b.jpg"]
    # c has no exif date, its mtime 1_000_000 is in 1970, so it sorts first
    assert names(list_images(tmp_path, sort="exif")) == ["c.jpg", "b.jpg", "a.jpg"]
    with pytest.raises(ValueError):
        list_images(tmp_path, sort="size")
    assert SORT_ORDERS[0] == "name"


def test_list_images_recursive_skips_hidden_dirs(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").mkdir()
    save(tmp_path / "z.png")
    save(tmp_path / "sub" / "a.jpg")
    save(tmp_path / ".hidden" / "b.jpg")
    flat = [p.relative_to(tmp_path).as_posix() for p in list_images(tmp_path)]
    deep = [p.relative_to(tmp_path).as_posix() for p in list_images(tmp_path, recursive=True)]
    assert flat == ["z.png"]
    assert deep == ["sub/a.jpg", "z.png"]
