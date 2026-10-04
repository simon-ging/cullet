"""
Rotate single video files with ffmpeg.

Takes the files to rotate as arguments instead of a directory, for the cases where
cullet-fix-rotation-videos guessed wrong or was unsure. The rotation is given by hand
with -R and applied to every file.

By default the rotated video is written next to the source as NAME_rot90.EXT and the source is
kept, with -O the source is replaced. Only the display rotation of the container is rewritten
and the streams are copied, which is instant, lossless and keeps the file the format it was.
With -P the rotated pixels are re-encoded to hevc in mp4 instead, so the extension can change.
A container that cannot store a rotation, avi and mpeg-ts, is refused unless -P or -A is given,
where -A re-encodes only those files. The modification time of the source is kept.

Examples:
    # dry run, prints what it would do
    cullet-rotate-videos video.mp4 -R 90
    # write video_rot90.mp4 next to it
    cullet-rotate-videos video.mp4 -R 90 -w
    # rotate several files in place, re-encoding the pixels
    cullet-rotate-videos a.mp4 b.mov -R 180 -P -O -w
"""

import logging
from pathlib import Path
from typing import Optional

from attrs import define
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from cullet.logs import configure_logging
from cullet.video import (
    DEFAULT_CRF,
    DEFAULT_PRESET,
    ROTATE_FILTERS,
    execute_video_rotations,
    plan_video_rotations,
)

logger = logging.getLogger(__name__)


@define
class Args(VerboseQuietArgs):
    input_files: list[Path] = add_argument(
        positional=True, type=str, nargs="+", help="Video files to rotate"
    )
    rotate: Optional[int] = add_argument(
        shortcut="-R",
        type=int,
        help=f"Rotate clockwise by this many degrees, one of {sorted(ROTATE_FILTERS)}.",
    )
    crf: int = add_argument(
        shortcut="-Q",
        type=int,
        default=DEFAULT_CRF,
        help="Encoder CRF quality for -P, lower is better and bigger.",
    )
    preset: str = add_argument(type=str, default=DEFAULT_PRESET, help="Encoder preset for -P.")
    rotate_pixels: bool = add_argument(
        shortcut="-P",
        action="store_true",
        help="Re-encode the rotated pixels of every file instead of only rewriting the display "
        "rotation of the container.",
    )
    rotate_pixels_if_needed: bool = add_argument(
        shortcut="-A",
        action="store_true",
        help="Re-encode the rotated pixels only for the containers that cannot store a rotation "
        "(avi, mpeg-ts) and rewrite the container rotation for all the others.",
    )
    overwrite: bool = add_argument(
        shortcut="-O", action="store_true", help="Overwrite the source instead of writing a copy"
    )
    write: bool = add_argument(shortcut="-w", action="store_true", help="Write changes to disk")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")
    if args.rotate is None:
        raise ValueError(f"Specify the rotation with -R, one of {sorted(ROTATE_FILTERS)}.")
    requests = []
    for input_file in args.input_files:
        src_file = Path(input_file)
        out_file = (
            src_file
            if args.overwrite
            else src_file.with_name(f"{src_file.stem}_rot{args.rotate}{src_file.suffix}")
        )
        requests.append((src_file, out_file, args.rotate))
    rotations = plan_video_rotations(
        requests,
        rotate_pixels=args.rotate_pixels,
        rotate_pixels_if_needed=args.rotate_pixels_if_needed,
        delete_source=args.overwrite,
    )
    execute_video_rotations(rotations, write=args.write, crf=args.crf, preset=args.preset)


if __name__ == "__main__":
    main()
