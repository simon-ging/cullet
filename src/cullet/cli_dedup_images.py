"""
Find duplicate images regardless of their size, using the SSCD copy detection model.

In each group of duplicates one image is kept: the one with at least 5% more pixels, else the
one with at least 20% bigger file size (see --tolerance_pixels and --tolerance_size), else
the first in name order. The other images are moved to the trash, or to a quarantine dir with
-Q, or deleted permanently with --unlink.
Embeddings are cached per input dir, so rerunning after adding images only embeds the new ones.

Example: cullet-dedup-images /path/to/images -Q /path/to/quarantine -w

To review the groups by hand instead, open the viewer with the same settings prefixed:
cullet /path/to/images --dedup --dedup_threshold 0.9
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from attrs import define
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from cullet.dedup.files import PathSpecArgs, remove_duplicates, write_json
from cullet.dedup.image_groups import DedupImageArgs, dedup_config_from_args, describe
from cullet.extras import require_full_extra
from cullet.logs import configure_logging

logger = logging.getLogger(__name__)


@define
class Args(DedupImageArgs, PathSpecArgs, VerboseQuietArgs):
    input_dir: Optional[Path] = add_argument(positional=True, type=str, help="Input image dir")
    quarantine_dir: Optional[Path] = add_argument(
        shortcut="-Q", type=str, help="Move duplicates here instead of deleting them."
    )
    report_file: Optional[Path] = add_argument(
        type=str, help="Write the duplicate groups with similarities to this json file."
    )
    delete_only: list[str] = add_argument(
        default=[],
        action="append",
        help="Gitignore-style pattern, only duplicates matching it are removed. Repeatable.",
    )
    unlink: bool = add_argument(
        action="store_true",
        help="Delete permanently instead of moving to the trash. This cannot be undone.",
    )
    write: bool = add_argument(shortcut="-w", action="store_true", help="Write changes to disk")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")
    require_full_extra()
    # torch is only loaded behind the check, so a missing extra gives a message that helps
    from cullet.dedup.images import find_duplicate_images

    input_dir = Path(args.input_dir)
    quarantine_dir = None if args.quarantine_dir is None else Path(args.quarantine_dir)

    result = find_duplicate_images(
        input_dir, dedup_config_from_args(args), pathspec_args=args, exclude_dir=quarantine_dir
    )
    logger.info(
        f"Found {result.n_pairs} similar pairs forming {len(result.groups)} duplicate groups."
    )
    report = []
    to_remove = []
    for group in result.groups:
        logger.info(f"KEEP   {describe(group.keep)} {group.keep.rel_file}")
        group_report = {"keep": group.keep.rel_file, "remove": []}
        for member in group.removals:
            logger.info(f"  DEL  {describe(member)} sim={member.sim:.3f} {member.rel_file}")
            group_report["remove"].append({"file": member.rel_file, "sim": member.sim})
            to_remove.append(member.rel_file)
        report.append(group_report)
    if args.report_file is not None:
        write_json(report, Path(args.report_file))
    if len(result.broken_files) > 0:
        logger.warning(
            f"Skipped {len(result.broken_files)} unreadable files: {list(result.broken_files)[:5]}"
        )
    actions = Counter()
    n_grouped = sum(len(group.members) for group in result.groups)
    actions["unique"] += result.n_images - n_grouped
    actions["duplicate_keep"] += len(result.groups)
    actions["duplicate_remove"] += len(to_remove)
    actions["unreadable"] += len(result.broken_files)
    summary = {k: v for k, v in actions.most_common() if v}
    logger.info(
        f"Images: {result.n_images}, duplicate groups: {len(result.groups)}, " f"actions: {summary}"
    )
    remove_duplicates(
        input_dir, to_remove, quarantine_dir, args.write, args.delete_only, args.unlink
    )


if __name__ == "__main__":
    main()
