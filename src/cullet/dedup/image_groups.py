"""
Settings and result types of the image deduplication, without torch so the viewer can import
them cheaply. The detection settings are defined once in DEDUP_IMAGE_ARG_SPECS and turned into
typedparser argument classes, so the CLI and the viewer accept the same options (the viewer
with a prefix, e.g. --dedup_threshold).
"""

from pathlib import Path
from typing import Any

from attrs import define, make_class

from typedparser import add_argument

# name -> (python type, add_argument kwargs). Shortcuts are only used in the unprefixed class.
DEDUP_IMAGE_ARG_SPECS: dict[str, tuple[type, dict[str, Any]]] = {
    "non_recursive": (
        bool,
        dict(shortcut="-r", action="store_true", help="Do not search subdirectories for images."),
    ),
    "threshold": (
        float,
        dict(
            shortcut="-t",
            type=float,
            default=0.9,
            help="Cosine similarity of SSCD embeddings at which images count as duplicates. "
            "Resized copies score above 0.93, burst shots ~0.88, crops ~0.85, rotations ~0.8.",
        ),
    ),
    "batch_size": (int, dict(shortcut="-b", type=int, default=32, help="Model batch size")),
    "workers": (
        int,
        dict(shortcut="-j", type=int, default=4, help="Parallel workers for loading images."),
    ),
    "device": (
        str | None,
        dict(type=str, help="Torch device, default: cuda if available, else cpu"),
    ),
    "model_file": (
        Path | None,
        dict(
            shortcut="-m",
            type=str,
            help="SSCD torchscript file, default: downloaded to cache dir.",
        ),
    ),
    "reset_cache": (bool, dict(action="store_true", help="Recompute all embeddings.")),
    "tolerance_pixels": (
        float,
        dict(
            type=float,
            default=0.05,
            help="The image with more pixels is kept only if it has at least this fraction more.",
        ),
    ),
    "tolerance_size": (
        float,
        dict(
            type=float,
            default=0.2,
            help="The bigger file is kept only if it is at least this fraction bigger. A re-encode "
            "can easily be 10%% bigger without looking any better.",
        ),
    ),
}


def make_dedup_image_args_class(name: str, prefix: str = "") -> type:
    """
    Build an attrs mixin with one typedparser argument per detection setting. With a prefix the
    fields are renamed (threshold -> dedup_threshold) and the shortcuts dropped, so the class can
    be mixed into another program's arguments without collisions.
    """
    fields = {}
    for arg_name, (arg_type, kwargs) in DEDUP_IMAGE_ARG_SPECS.items():
        kwargs = dict(kwargs)
        if prefix:
            kwargs.pop("shortcut", None)
        attr = add_argument(**kwargs)
        attr.type = arg_type
        fields[f"{prefix}{arg_name}"] = attr
    return make_class(name, fields, slots=False)


DedupImageArgs = make_dedup_image_args_class("DedupImageArgs")


@define
class DedupImageConfig:
    """Detection settings, independent of how the results are acted upon."""

    recursive: bool = True
    threshold: float = 0.9
    batch_size: int = 32
    workers: int = 4
    device: str | None = None  # None: cuda if available
    model_file: Path | None = None
    reset_cache: bool = False
    tolerance_pixels: float = 0.05
    tolerance_size: float = 0.2


def dedup_config_from_args(args: Any, prefix: str = "") -> DedupImageConfig:
    """Read the detection settings from a parsed args instance built with the same prefix."""
    values = {name: getattr(args, f"{prefix}{name}") for name in DEDUP_IMAGE_ARG_SPECS}
    model_file = values.pop("model_file")
    non_recursive = values.pop("non_recursive")
    return DedupImageConfig(
        recursive=not non_recursive,
        model_file=None if model_file is None else Path(model_file),
        **values,
    )


@define
class DuplicateMember:
    path: Path  # absolute
    rel_file: str  # relative to the input dir, as in the CLI log
    width: int
    height: int
    size: int  # bytes
    sim: float  # similarity to the closest other member, that is what put it into the group


@define
class DuplicateGroup:
    members: list[DuplicateMember]  # the proposed keep first, then the rest in name order

    @property
    def keep(self) -> DuplicateMember:
        return self.members[0]

    @property
    def removals(self) -> list[DuplicateMember]:
        return self.members[1:]


@define
class DedupImageResult:
    groups: list[DuplicateGroup]
    n_images: int
    n_pairs: int
    broken_files: dict[str, str]  # rel_file -> error


def describe(member: DuplicateMember) -> str:
    """Fixed-width dimensions and file size, e.g. ' 4000x1440    1.38MB'."""
    return f"{member.width:5d}x{member.height:<5d} {member.size / 1024**2:6.2f}MB"
