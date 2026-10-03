"""
Image decoding and lossless on-disk rotation for the viewer.

The EXIF orientation tag is applied when decoding, so the viewer shows the picture the way the
camera intended. A rotation requested by the user is applied to the pixels and the file is
rewritten without EXIF, so afterwards no program can disagree about which way is up.
JPEGs are transformed losslessly with jpegtran (the ICC profile is kept), PNGs are re-encoded.
"""

import functools
import logging
import os
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path

from PIL import Image, ImageOps
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# pillow refuses images above 2x this limit as decompression bombs, panoramas are bigger than that
Image.MAX_IMAGE_PIXELS = 10 * Image.MAX_IMAGE_PIXELS
EXIF_ORIENTATION_TAG = 0x0112
EXIF_MODEL_TAG = 0x0110
EXIF_DATETIME_TAG = 0x0132
EXIF_IFD_TAG = 0x8769
EXIF_EXPOSURE_TIME_TAG = 0x829A
EXIF_FNUMBER_TAG = 0x829D
EXIF_ISO_TAG = 0x8827
EXIF_DATETIME_ORIGINAL_TAG = 0x9003
EXIF_FOCAL_LENGTH_TAG = 0x920A


class Transform(str, Enum):
    """The 8 symmetries of a rectangle. Rotations are clockwise as seen by the viewer."""

    IDENTITY = "identity"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_H = "flip_h"
    FLIP_V = "flip_v"
    TRANSPOSE = "transpose"
    TRANSVERSE = "transverse"


# how to display the stored pixels for each EXIF orientation value, same as ImageOps.exif_transpose
EXIF_ORIENTATION_TO_TRANSFORM = {
    1: Transform.IDENTITY,
    2: Transform.FLIP_H,
    3: Transform.ROTATE_180,
    4: Transform.FLIP_V,
    5: Transform.TRANSPOSE,
    6: Transform.ROTATE_90,
    7: Transform.TRANSVERSE,
    8: Transform.ROTATE_270,
}

# Pillow's ROTATE_90 is counter-clockwise
PIL_TRANSPOSE = {
    Transform.ROTATE_90: Image.Transpose.ROTATE_270,
    Transform.ROTATE_180: Image.Transpose.ROTATE_180,
    Transform.ROTATE_270: Image.Transpose.ROTATE_90,
    Transform.FLIP_H: Image.Transpose.FLIP_LEFT_RIGHT,
    Transform.FLIP_V: Image.Transpose.FLIP_TOP_BOTTOM,
    Transform.TRANSPOSE: Image.Transpose.TRANSPOSE,
    Transform.TRANSVERSE: Image.Transpose.TRANSVERSE,
}

JPEGTRAN_ARGS = {
    Transform.IDENTITY: [],
    Transform.ROTATE_90: ["-rotate", "90"],
    Transform.ROTATE_180: ["-rotate", "180"],
    Transform.ROTATE_270: ["-rotate", "270"],
    Transform.FLIP_H: ["-flip", "horizontal"],
    Transform.FLIP_V: ["-flip", "vertical"],
    Transform.TRANSPOSE: ["-transpose"],
    Transform.TRANSVERSE: ["-transverse"],
}

Grid = tuple[tuple[int, ...], ...]

# the same transforms on a grid of rows, used to compose them
GRID_OPS = {
    Transform.IDENTITY: lambda g: g,
    Transform.ROTATE_90: lambda g: tuple(zip(*g[::-1])),
    Transform.ROTATE_180: lambda g: tuple(row[::-1] for row in g[::-1]),
    Transform.ROTATE_270: lambda g: tuple(zip(*g))[::-1],
    Transform.FLIP_H: lambda g: tuple(row[::-1] for row in g),
    Transform.FLIP_V: lambda g: g[::-1],
    Transform.TRANSPOSE: lambda g: tuple(zip(*g)),
    Transform.TRANSVERSE: lambda g: tuple(zip(*(row[::-1] for row in g[::-1]))),
}

# small asymmetric grid where every transform gives a different result
_PROBE: Grid = ((0, 1, 2), (3, 4, 5))


def identify_transform(transformed_probe: Grid) -> Transform:
    for transform, op in GRID_OPS.items():
        if op(_PROBE) == transformed_probe:
            return transform
    raise ValueError(f"Grid is not a transform of the probe: {transformed_probe}")


@functools.lru_cache(maxsize=None)
def compose(first: Transform, then: Transform) -> Transform:
    """The single transform equal to applying first, then the second one."""
    return identify_transform(GRID_OPS[then](GRID_OPS[first](_PROBE)))


def inverse(transform: Transform) -> Transform:
    for candidate in Transform:
        if compose(transform, candidate) == Transform.IDENTITY:
            return candidate
    raise ValueError(f"No inverse found for {transform}")


def get_exif_orientation(img: Image.Image) -> int:
    orientation = img.getexif().get(EXIF_ORIENTATION_TAG, 1)
    if orientation not in EXIF_ORIENTATION_TO_TRANSFORM:
        # some cameras write 0 or garbage, treat as unrotated like other viewers do
        return 1
    return orientation


def exif_datetime(exif: Image.Exif) -> str | None:
    """Date taken as 'YYYY:MM:DD HH:MM:SS', from DateTimeOriginal or the plain DateTime tag."""
    date = exif.get_ifd(EXIF_IFD_TAG).get(EXIF_DATETIME_ORIGINAL_TAG) or exif.get(EXIF_DATETIME_TAG)
    return str(date) if date else None


def read_exif_datetime(path: Path) -> str | None:
    with Image.open(path) as img:
        return exif_datetime(img.getexif())


def read_exif_summary(path: Path) -> str:
    """One line for the info overlay: date taken, camera, exposure, file size. Empty parts
    are left out. Without an EXIF date the file's modification time is shown instead."""
    with Image.open(path) as img:
        exif = img.getexif()
        ifd = exif.get_ifd(EXIF_IFD_TAG)
    stat = path.stat()
    parts = []
    date = exif_datetime(exif)
    if date:
        parts.append(date)
    else:
        parts.append(f"mtime {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M:%S}")
    model = exif.get(EXIF_MODEL_TAG)
    if model:
        parts.append(str(model).strip())
    exposure = ifd.get(EXIF_EXPOSURE_TIME_TAG)
    if exposure:
        exposure = float(exposure)
        parts.append(f"1/{round(1 / exposure)}s" if exposure < 1 else f"{exposure:g}s")
    fnumber = ifd.get(EXIF_FNUMBER_TAG)
    if fnumber:
        parts.append(f"f/{float(fnumber):.1f}")
    iso = ifd.get(EXIF_ISO_TAG)
    if iso:
        parts.append(f"ISO {int(iso)}")
    focal = ifd.get(EXIF_FOCAL_LENGTH_TAG)
    if focal:
        parts.append(f"{float(focal):.0f}mm")
    parts.append(f"{stat.st_size / 1e6:.1f} MB")
    return "  ".join(parts)


def load_image(path: Path, max_side: int | None = None) -> Image.Image:
    """
    Decode an image with the EXIF orientation applied.

    Args:
        path: jpg or png file
        max_side: if given, JPEGs are decoded at a reduced DCT scale that still covers this
            size on both sides, which is many times faster for thumbnails. The result can be
            larger than max_side, shrink it afterwards.
    """
    img = Image.open(path)
    if max_side is not None:
        img.draft(img.mode, (max_side, max_side))
    img = ImageOps.exif_transpose(img)
    img.load()
    return img


def load_thumbnail(path: Path, size: int) -> Image.Image:
    img = load_image(path, max_side=size)
    img.thumbnail((size, size))
    return img


def pil_to_qimage(img: Image.Image) -> QImage:
    if img.mode == "RGB":
        qt_format, channels = QImage.Format.Format_RGB888, 3
    else:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        qt_format, channels = QImage.Format.Format_RGBA8888, 4
    data = img.tobytes("raw", img.mode)
    qimage = QImage(data, img.width, img.height, img.width * channels, qt_format)
    # QImage does not own the buffer, so copy before the bytes go out of scope
    return qimage.copy()


def rotate_file(path: Path, transform: Transform) -> None:
    """
    Apply the transform to the picture as displayed and write the file back without EXIF.

    The stored EXIF orientation is folded into the transform, so the result equals the
    displayed image after the rotation, no matter how the pixels were stored before.
    """
    path = Path(path)
    with Image.open(path) as img:
        stored = EXIF_ORIENTATION_TO_TRANSFORM[get_exif_orientation(img)]
        fmt = img.format
    total = compose(stored, transform)
    tmp_file = path.with_name(f"{path.name}.rotating")
    if fmt == "JPEG":
        _jpegtran(path, tmp_file, total)
    elif fmt == "PNG":
        _rewrite_png(path, tmp_file, total)
    else:
        raise ValueError(f"Cannot rotate {path}, unsupported format {fmt}")
    os.replace(tmp_file, path)
    logger.info(f"Rotated {path} ({stored.value} then {transform.value} = {total.value})")


def _jpegtran(src: Path, dst: Path, transform: Transform) -> None:
    base_cmd = ["jpegtran", "-copy", "icc", *JPEGTRAN_ARGS[transform]]
    io_args = ["-outfile", dst.as_posix(), src.as_posix()]
    result = subprocess.run([*base_cmd, "-perfect", *io_args], capture_output=True, text=True)
    if result.returncode != 0:
        # -perfect refuses when width or height is not a multiple of the MCU size (8 or 16 px),
        # because the partial edge blocks cannot be transformed exactly. -trim drops those edge
        # pixels instead, which keeps every remaining pixel exact.
        logger.warning(
            f"Rotating {src} is not exact at the edges, trimming partial blocks: "
            f"{result.stderr.strip()}"
        )
        result = subprocess.run([*base_cmd, "-trim", *io_args], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"jpegtran failed for {src}: {result.stderr.strip()}")


def _rewrite_png(src: Path, dst: Path, transform: Transform) -> None:
    with Image.open(src) as img:
        icc_profile = img.info.get("icc_profile")
        if transform == Transform.IDENTITY:
            out = img.copy()
        else:
            out = img.transpose(PIL_TRANSPOSE[transform])
    out.save(dst, format="PNG", icc_profile=icc_profile)
