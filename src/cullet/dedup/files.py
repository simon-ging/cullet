"""
The file side of the deduplication, without torch: index a folder, filter paths with
gitignore-style or regex patterns, and remove the duplicates.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from attrs import define
from natsort import natsorted
from pathspec import PathSpec, RegexPattern
from typedparser import add_argument

from cullet.file_ops import move_to_system_trash

logger = logging.getLogger(__name__)


@define(slots=False)
class PathSpecArgs:
    exclude_git: list[str] = add_argument(
        shortcut="-x", default=[], action="append", help="Git-like pathspec list to exclude files."
    )
    exclude_regex: list[str] = add_argument(
        shortcut="-X", default=[], action="append", help="Regex pathspec list to exclude files."
    )
    include_git: list[str] = add_argument(
        shortcut="-i", default=[], action="append", help="Git-like pathspec list to include files."
    )
    include_regex: list[str] = add_argument(
        shortcut="-I", default=[], action="append", help="Regex pathspec list to include files."
    )
    exclude_gitignore_file: Optional[Path] = add_argument(
        shortcut="-g", type=str, default=None, help="Gitignore file for matching files"
    )


def filter_rel_files(
    rel_files: list[str],
    include_git: list[str] | None = None,
    include_regex: list[str] | None = None,
    exclude_git: list[str] | None = None,
    exclude_regex: list[str] | None = None,
    exclude_gitignore_file: Path | None = None,
) -> list[str]:
    """
    Keep the files that match every include list and no exclude list. Empty lists do nothing.

    Args:
        rel_files: files relative to the folder the patterns are anchored at
        include_git: gitignore-style patterns, a file has to match one of them
        include_regex: regular expressions, a file has to match one of them
        exclude_git: gitignore-style patterns, a file matching one of them is dropped
        exclude_regex: regular expressions, a file matching one of them is dropped
        exclude_gitignore_file: a file of gitignore-style patterns to exclude
    """
    if exclude_gitignore_file is not None:
        lines = Path(exclude_gitignore_file).read_text(encoding="utf-8").splitlines()
        exclude_git = list(exclude_git or []) + lines
    for patterns, syntax, negate in [
        (include_git, "gitignore", False),
        (include_regex, RegexPattern, False),
        (exclude_git, "gitignore", True),
        (exclude_regex, RegexPattern, True),
    ]:
        if not patterns:
            continue
        spec = PathSpec.from_lines(syntax, patterns)
        # a gitignore pattern with a leading slash is anchored at the root, so the files get one
        kept = set(spec.match_files([f"/{f}" for f in rel_files], negate=negate))
        rel_files = [f for f in rel_files if f"/{f}" in kept]
    return rel_files


def index_files(
    input_dir: Path, recursive: bool, pathspec_args: PathSpecArgs | None = None
) -> dict[str, os.stat_result]:
    """All files of the folder with their stat, keyed by the path relative to the folder."""
    rel_files = []
    for root, dirs, names in os.walk(input_dir):
        rel_files.extend((Path(root) / name).relative_to(input_dir).as_posix() for name in names)
        if not recursive:
            dirs.clear()
    if pathspec_args is not None:
        rel_files = filter_rel_files(
            rel_files,
            include_git=pathspec_args.include_git,
            include_regex=pathspec_args.include_regex,
            exclude_git=pathspec_args.exclude_git,
            exclude_regex=pathspec_args.exclude_regex,
            exclude_gitignore_file=pathspec_args.exclude_gitignore_file,
        )
    return {rel_file: (input_dir / rel_file).stat() for rel_file in rel_files}


def write_json(data, json_file: Path) -> None:
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def remove_duplicates(
    base_dir: Path,
    rel_files: list[str],
    quarantine_dir: Path | None,
    write: bool,
    delete_only: list[str] | None = None,
    unlink: bool = False,
):
    """
    Remove the files, by default into the system trash so a wrong call can be undone.

    Args:
        base_dir: the files are relative to this dir
        rel_files: files to remove
        quarantine_dir: move the files here keeping their relative path, instead of trashing
        write: apply the changes, otherwise only log them
        delete_only: gitignore-style patterns, if given only matching files are removed
        unlink: delete permanently instead of trashing, which cannot be undone

    Raises:
        OSError: when the filesystem of a file has no trash. Pass a quarantine dir or unlink
            instead of falling back to a silent permanent delete.
    """
    if delete_only:
        kept = filter_rel_files(rel_files, include_git=delete_only)
        logger.info(f"Filter {delete_only} keeps {len(kept)} of {len(rel_files)} files to remove")
        rel_files = kept
    verb = "" if write else "Would "
    for rel_file in natsorted(rel_files):
        src_file = (base_dir / rel_file).absolute()
        if quarantine_dir is None:
            logger.info(f"{verb}{'Delete' if unlink else 'Trash'} {src_file}")
            if write:
                if unlink:
                    src_file.unlink()
                else:
                    move_to_system_trash(src_file)
            continue
        tgt_file = quarantine_dir / rel_file
        logger.info(f"{verb}Move {src_file} -> {tgt_file}")
        if write:
            assert not tgt_file.exists(), f"Quarantine target {tgt_file} exists already"
            tgt_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_file, tgt_file)
    if not write:
        logger.warning("Dry run. Use -w to apply the changes.")
