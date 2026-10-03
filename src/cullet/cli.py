"""
Photo viewer. Open a folder, or an image file to start at that image.

    cullet /path/to/photos

With --dedup the folder is searched for duplicate images first (needs the dedup extra, the
settings are the dedup_ options) and the viewer opens the groups for review: keys 1-9 trash one
member, P accepts the proposal, J/K step through the groups.

    cullet /path/to/photos --dedup --dedup_threshold 0.88

With target folders the keys 1-9 move the current image into the folder of that number, in the
order they are given. That uses the same keys as the review, so it cannot be combined with
--dedup.

    cullet /path/to/photos -a ../good -a ../maybe

Tags work the same way and share the number keys: they take the keys behind the target folders,
and pressing one adds the tag to the file name or removes it if it is there already.

    cullet /path/to/photos -a ../good -T sun -T portrait
"""

import logging
import sys
from pathlib import Path

from attrs import define
from PySide6.QtWidgets import QApplication

from typedparser import TypedParser, VerboseQuietArgs, add_argument

from cullet.dedup_review import DedupReview
from cullet.file_ops import TAG_MARKER
from cullet.main_window import DEFAULT_TRASH_DIR, MainWindow
from cullet.thumb_store import reset_thumb_stores
from cullet.dedup.image_groups import dedup_config_from_args, make_dedup_image_args_class

logger = logging.getLogger(__name__)
VERBOSITY_LADDER = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
DEDUP_PREFIX = "dedup_"
DedupPrefixedArgs = make_dedup_image_args_class("DedupPrefixedArgs", prefix=DEDUP_PREFIX)


@define
class Args(DedupPrefixedArgs, VerboseQuietArgs):
    path: Path = add_argument(
        positional=True, type=str, help="Folder of images, or an image file to start at"
    )
    trash_dir: Path = add_argument(
        type=str,
        default=DEFAULT_TRASH_DIR.as_posix(),
        help="Deleted files are moved here, mirroring their absolute path",
    )
    target_folder: list[str] = add_argument(
        shortcut="-a",
        type=str,
        default=[],
        action="append",
        help="Folder the keys 1-9 move images into, repeat for up to 9 folders",
    )
    tag: list[str] = add_argument(
        shortcut="-T",
        type=str,
        default=[],
        action="append",
        help="Tag toggled by the number key behind the target folder keys, repeat for more tags",
    )
    reset_cache: bool = add_argument(
        action="store_true",
        help="Delete the stored thumbnails of the folder and its subfolders, and with --dedup "
        "recompute the embeddings. Needed after editing images without changing their mtime.",
    )
    dedup: bool = add_argument(
        action="store_true",
        help="Find duplicate images in the folder and open them for review, with the "
        "dedup_ options as settings. Needs the dedup extra installed.",
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    logging.basicConfig(
        level=get_log_level(args),
        format="%(asctime)s %(levelname).4s %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    logger.info(f"{args}")
    path = Path(args.path).absolute()
    if path.is_file():
        folder, start_file = path.parent, path
    elif path.is_dir():
        folder, start_file = path, None
    else:
        raise FileNotFoundError(f"{path} is neither a folder nor a file")

    target_dirs = [Path(target).absolute() for target in args.target_folder]
    tags = list(args.tag)
    if (target_dirs or tags) and args.dedup:
        raise ValueError(
            "--target_folder, --tag and --dedup all bind the keys 1-9, use one at a time"
        )
    for tag in tags:
        if len(tag.split()) != 1 or "/" in tag or TAG_MARKER in tag:
            raise ValueError(f"Tag {tag!r} must be one word without a '/' or a '{TAG_MARKER}'")
    if len(target_dirs) + len(tags) > 9:
        raise ValueError(
            f"There are 9 number keys, {len(target_dirs)} target folders and {len(tags)} tags "
            f"do not fit"
        )

    if args.reset_cache:
        reset_thumb_stores(folder)

    review = None
    if args.dedup:
        # torch is only loaded when needed, it costs seconds at startup
        from cullet.dedup.images import find_duplicate_images

        config = dedup_config_from_args(args, DEDUP_PREFIX)
        config.reset_cache = config.reset_cache or args.reset_cache
        result = find_duplicate_images(folder, config)
        if len(result.groups) == 0:
            logger.warning(f"No duplicates in {folder}, opening it as a plain folder")
        else:
            review = DedupReview(result)
            logger.info(review.summary())

    app = QApplication(sys.argv[:1])
    window = MainWindow(
        folder,
        start_file=start_file,
        trash_dir=Path(args.trash_dir),
        review=review,
        target_dirs=target_dirs,
        tags=tags,
    )
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec())


def get_log_level(args: VerboseQuietArgs) -> str:
    """INFO by default, every -v or -q moves one step along the ladder."""
    if args.loglevel is not None:
        assert not (args.verbose or args.quiet), "Cannot set both -v/-q and --loglevel"
        assert args.loglevel in VERBOSITY_LADDER, f"{args.loglevel=} not in {VERBOSITY_LADDER}"
        return args.loglevel
    index = VERBOSITY_LADDER.index("INFO") + args.verbose - args.quiet
    return VERBOSITY_LADDER[max(0, min(len(VERBOSITY_LADDER) - 1, index))]
