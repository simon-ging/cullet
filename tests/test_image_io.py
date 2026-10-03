import shutil

import numpy as np
import pytest
from PIL import Image

from cullet.image_io import (
    EXIF_ORIENTATION_TAG,
    EXIF_ORIENTATION_TO_TRANSFORM,
    GRID_OPS,
    PIL_TRANSPOSE,
    Transform,
    compose,
    get_exif_orientation,
    inverse,
    load_image,
    load_thumbnail,
    rotate_file,
)

# the transforms on a numpy array with axes (height, width, ...), the reference for GRID_OPS
NUMPY_OPS = {
    Transform.IDENTITY: lambda a: a,
    Transform.ROTATE_90: lambda a: np.rot90(a, k=-1),
    Transform.ROTATE_180: lambda a: np.rot90(a, k=2),
    Transform.ROTATE_270: lambda a: np.rot90(a, k=1),
    Transform.FLIP_H: lambda a: a[:, ::-1],
    Transform.FLIP_V: lambda a: a[::-1, :],
    Transform.TRANSPOSE: lambda a: np.swapaxes(a, 0, 1),
    Transform.TRANSVERSE: lambda a: np.swapaxes(a[::-1, ::-1], 0, 1),
}

# blocks of flat color survive JPEG compression almost unchanged, so the pixel comparison
# below can use a small tolerance. 48x32 is a multiple of the 16 px MCU, 50x30 is not.
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


def make_displayed_image(width: int, height: int) -> np.ndarray:
    """Asymmetric picture in the orientation the photographer intended: 3x2 colored blocks."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    block_h, block_w = height // 2, width // 3
    for row in range(2):
        for col in range(3):
            color = COLORS[row * 3 + col]
            arr[row * block_h : (row + 1) * block_h, col * block_w : (col + 1) * block_w] = color
    return arr


def save_with_orientation(arr: np.ndarray, path, orientation: int, **save_kwargs) -> None:
    """Store the pixels such that a viewer honoring the EXIF tag shows arr."""
    stored = NUMPY_OPS[inverse(EXIF_ORIENTATION_TO_TRANSFORM[orientation])](arr)
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = orientation
    if str(path).endswith(".jpg"):
        # no chroma subsampling, so the block edges of the test picture stay sharp
        save_kwargs = {"quality": 95, "subsampling": 0, **save_kwargs}
    Image.fromarray(np.ascontiguousarray(stored)).save(path, exif=exif.tobytes(), **save_kwargs)


def assert_images_close(actual: np.ndarray, expected: np.ndarray, tol: float):
    assert actual.shape == expected.shape, f"{actual.shape} != {expected.shape}"
    diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64)).mean()
    assert diff <= tol, f"Mean abs pixel difference {diff} > {tol}"


@pytest.mark.parametrize("transform", list(Transform))
def test_grid_ops_match_numpy(transform):
    probe = np.arange(24).reshape(4, 6)
    grid = tuple(tuple(row) for row in probe.tolist())
    expected = NUMPY_OPS[transform](probe).tolist()
    assert [list(row) for row in GRID_OPS[transform](grid)] == expected


@pytest.mark.parametrize("first", list(Transform))
@pytest.mark.parametrize("then", list(Transform))
def test_compose_matches_sequential_application(first, then):
    probe = np.arange(24).reshape(4, 6)
    sequential = NUMPY_OPS[then](NUMPY_OPS[first](probe))
    composed = NUMPY_OPS[compose(first, then)](probe)
    assert np.array_equal(sequential, composed)


@pytest.mark.parametrize("transform", list(Transform))
def test_inverse(transform):
    assert compose(transform, inverse(transform)) == Transform.IDENTITY


@pytest.mark.parametrize("transform", [t for t in Transform if t != Transform.IDENTITY])
def test_pil_transpose_matches_numpy(transform):
    arr = make_displayed_image(48, 32)
    via_pil = np.asarray(Image.fromarray(arr).transpose(PIL_TRANSPOSE[transform]))
    assert np.array_equal(via_pil, NUMPY_OPS[transform](arr))


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("suffix,tol", [(".jpg", 2.0), (".png", 0.0)])
def test_load_applies_exif_orientation(tmp_path, orientation, suffix, tol):
    displayed = make_displayed_image(48, 32)
    path = tmp_path / f"img{suffix}"
    save_with_orientation(displayed, path, orientation)
    with Image.open(path) as img:
        assert get_exif_orientation(img) == orientation
    assert_images_close(np.asarray(load_image(path)), displayed, tol)


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("suffix,tol", [(".jpg", 2.0), (".png", 0.0)])
@pytest.mark.parametrize("turn", [Transform.ROTATE_90, Transform.ROTATE_270])
def test_rotate_file_folds_exif_and_strips_it(tmp_path, orientation, suffix, tol, turn):
    if shutil.which("jpegtran") is None and suffix == ".jpg":
        pytest.skip("jpegtran not installed")
    displayed = make_displayed_image(48, 32)
    path = tmp_path / f"img{suffix}"
    save_with_orientation(displayed, path, orientation)

    rotate_file(path, turn)

    with Image.open(path) as img:
        assert EXIF_ORIENTATION_TAG not in img.getexif(), "EXIF orientation must be gone"
    assert_images_close(np.asarray(load_image(path)), NUMPY_OPS[turn](displayed), tol)


def test_rotate_jpeg_with_partial_mcu_blocks_trims(tmp_path):
    if shutil.which("jpegtran") is None:
        pytest.skip("jpegtran not installed")
    displayed = make_displayed_image(50, 30)
    path = tmp_path / "img.jpg"
    Image.fromarray(displayed).save(path)
    rotate_file(path, Transform.ROTATE_90)
    with Image.open(path) as img:
        width, height = img.size
    # the rotated image is 30x50 minus at most one partial block per edge
    assert 30 - 16 < width <= 30 and 50 - 16 < height <= 50


def test_rotate_keeps_icc_profile(tmp_path):
    if shutil.which("jpegtran") is None:
        pytest.skip("jpegtran not installed")
    path = tmp_path / "img.jpg"
    fake_icc = b"\x00" * 128 + b"fake icc profile"
    Image.fromarray(make_displayed_image(48, 32)).save(path, icc_profile=fake_icc)
    rotate_file(path, Transform.ROTATE_180)
    with Image.open(path) as img:
        assert img.info.get("icc_profile") == fake_icc


def test_thumbnail_draft_decode_is_oriented(tmp_path):
    displayed = make_displayed_image(480, 320)
    path = tmp_path / "img.jpg"
    save_with_orientation(displayed, path, 6)
    thumb = load_thumbnail(path, 64)
    assert max(thumb.size) == 64
    assert thumb.size[0] > thumb.size[1], "landscape picture must stay landscape"
    expected = np.asarray(Image.fromarray(displayed).resize(thumb.size))
    assert_images_close(np.asarray(thumb), expected, 6.0)
