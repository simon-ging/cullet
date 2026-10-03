"""
Where cullet keeps its files, and how a folder is named inside those places.
"""

import getpass
import os
import tempfile
import urllib.parse
from pathlib import Path

APP_NAME = "cullet"


def get_cache_dir() -> Path:
    """Cache dir for thumbnails and model weights, following the XDG base directory spec."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / APP_NAME


def get_temp_cache_dir() -> Path:
    """
    Cache dir for computed data, gone after a reboot.

    The temp dir is shared between users, so the name includes the user to avoid collisions.
    """
    return Path(tempfile.gettempdir()) / f"{APP_NAME}-{getpass.getuser()}" / "cache"


def folder_cache_name(folder: Path) -> str:
    """The absolute path of the folder as a single file name, used to key caches per folder."""
    return urllib.parse.quote(Path(folder).absolute().as_posix(), safe="")


def is_inside_dir(rel_file: str, base_dir: Path, other_dir: Path | None) -> bool:
    """
    Whether a file of one directory actually lies inside another one.

    Used to keep a scan from picking up what the same run writes, e.g. the output directory a
    tool creates inside the directory it reads.

    Args:
        rel_file: file relative to base_dir
        base_dir: the directory rel_file is relative to
        other_dir: the directory to test against, None means nothing is excluded

    Returns:
        whether the file is inside other_dir
    """
    if other_dir is None:
        return False
    return other_dir.absolute() in (base_dir / rel_file).absolute().parents
