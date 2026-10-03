"""
Downscale videos with ffmpeg to HEVC (H.265) mp4. Videos that are already small are skipped.

Example: cullet-downscale-videos /path/to/videos -s 1080 -w

TODO the temp file should either be in a different folder, or renamed somehing like .tmp at the end, so it doesnt stay and get processed itself, if this script crashes.
"""

import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional

from attrs import define
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from cullet.files import PathSpecArgs, index_files
from cullet.logs import configure_logging
from cullet.paths import is_inside_dir

logger = logging.getLogger(__name__)


@define
class Args(PathSpecArgs, VerboseQuietArgs):
    input_dir: Optional[Path] = add_argument(positional=True, type=str, help="Input video dir")
    non_recursive: bool = add_argument(
        shortcut="-r", action="store_true", help="Do not search subdirectories for videos."
    )
    smaller_side: int | None = add_argument(
        shortcut="-s",
        type=int,
        help="Resize videos so that the smaller side is this many pixels long.",
    )
    bigger_side: int | None = add_argument(
        shortcut="-b",
        type=int,
        help="Resize videos so that the larger side is this many pixels long.",
    )
    crf: int = add_argument(
        shortcut="-c", type=int, default=23, help="x265 CRF quality, lower is better and bigger."
    )
    preset: str = add_argument(
        shortcut="-p", type=str, default="slow", help="x265 preset, slower is smaller."
    )
    output_dir_rel: Path = add_argument(
        shortcut="-o",
        type=str,
        help="Output directory relative to input dir.",
        default="downscaled_output",
    )
    overwrite: bool = add_argument(
        shortcut="-O", action="store_true", help="Overwrite files instead of a separate output dir"
    )
    write: bool = add_argument(shortcut="-w", action="store_true", help="Write changes to disk")


ENDINGS = [".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".mts"]
OUTPUT_EXTENSION = ".mp4"
# unfinished outputs of interrupted runs, never used as input
TMP_SUFFIX = f".tmp{OUTPUT_EXTENSION}"


def probe_streams(video_file: Path) -> list[dict]:
    """
    Read the streams of a video.

    Raises:
        subprocess.CalledProcessError: when ffprobe refuses the file, e.g. an empty one
        ValueError: when ffprobe writes something that is not the expected json
    """
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            video_file.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    try:
        return json.loads(output)["streams"]
    except json.JSONDecodeError as e:
        raise ValueError(f"ffprobe wrote no usable json: {output[:200]!r}") from e


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")

    if int(args.smaller_side is not None) + int(args.bigger_side is not None) != 1:
        raise ValueError("Specify either --smaller_side or --bigger_side but not both.")

    # scaling by smaller or bigger side does not depend on the orientation, so the ffmpeg
    # expression works on the auto-rotated frames and the size check works on the raw stream
    if args.smaller_side is not None:
        target_side = args.smaller_side
        is_width = "lt(iw,ih)"
    else:
        target_side = args.bigger_side
        is_width = "gte(iw,ih)"
    scale_filter = f"scale=w='if({is_width},{target_side},-2)':h='if({is_width},-2,{target_side})'"

    src_index = index_files(Path(args.input_dir), not args.non_recursive, args)
    files = list(src_index.keys())
    logger.info(f"Found {len(files)} files.")
    # the output dir lives inside the input dir, so it has to be kept out of the index or a
    # second run would process what the first one wrote
    input_dir = Path(args.input_dir)
    output_dir = input_dir if args.overwrite else input_dir / args.output_dir_rel
    exclude_dir = None if args.overwrite else output_dir
    files2 = [
        f
        for f in files
        if any(f.lower().endswith(a) for a in ENDINGS)
        and not f.endswith(TMP_SUFFIX)
        and not is_inside_dir(f, input_dir, exclude_dir)
    ]
    logger.info(f"From those detected {len(files2)} videos.")
    # sources that could keep their name come first, so they get to claim it before the others
    files2.sort(key=lambda f: (not f.lower().endswith(OUTPUT_EXTENSION), f))
    source_files_dict = {f: None for f in files2}

    target2source = {}
    actions = Counter()
    unreadable = []
    total_size_before, total_size_after = 0, 0
    for src_file_rel in source_files_dict.keys():
        src_file_path = Path(args.input_dir) / src_file_rel
        # a directory of videos can hold empty or truncated files, those are reported and
        # skipped instead of stopping the whole run
        try:
            streams = probe_streams(src_file_path)
        except (subprocess.CalledProcessError, ValueError, KeyError) as e:
            error = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
            # ffprobe writes several lines, the last one names the actual problem
            error = (error or "").strip().splitlines()[-1] if error else repr(e)
            logger.error(f"Skipping {src_file_rel}: {error}")
            unreadable.append(src_file_rel)
            actions["unreadable"] += 1
            continue
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if len(video_streams) == 0:
            logger.error(f"Skipping {src_file_rel}: no video stream")
            unreadable.append(src_file_rel)
            actions["no_video_stream"] += 1
            continue
        width, height = video_streams[0].get("width"), video_streams[0].get("height")
        if not width or not height:
            logger.error(f"Skipping {src_file_rel}: the video stream reports no size")
            unreadable.append(src_file_rel)
            actions["no_video_stream"] += 1
            continue
        side = min(width, height) if args.smaller_side is not None else max(width, height)
        if side <= target_side:
            logger.debug(f"Skip {src_file_rel} -- already small ({width}x{height})")
            actions["already_small"] += 1
            continue

        old_extension = "." + src_file_rel.split(".")[-1].lower()
        is_new_extension = OUTPUT_EXTENSION != old_extension
        tgt_file_rel_stem = ".".join(src_file_rel.split(".")[:-1])
        tgt_file_rel = f"{tgt_file_rel_stem}{OUTPUT_EXTENSION}"

        # rename until no more conflicts
        for i in range(100):
            taken = False
            if tgt_file_rel in target2source:
                # same target twice, must be renamed
                taken = True
            if tgt_file_rel in source_files_dict and is_new_extension and args.overwrite:
                # target points to another source file, must be renamed
                taken = True
            if not taken:
                break
            tgt_file_rel_new = f"{tgt_file_rel_stem}-copy{i:02d}{OUTPUT_EXTENSION}"
            logger.warning(f"Target {tgt_file_rel} taken -> rename to {tgt_file_rel_new}")
            tgt_file_rel = tgt_file_rel_new
        else:
            raise ValueError(f"Could not find a free name for {src_file_rel} after 100 tries.")
        target2source[tgt_file_rel] = src_file_rel

        out_file = output_dir / tgt_file_rel
        if not args.overwrite and out_file.is_file():
            logger.debug(f"Skip {src_file_rel} -- output {out_file} already exists")
            actions["output_exists"] += 1
            continue

        # aac can be copied into mp4 without quality loss, everything else is converted to aac
        audio_codecs = [s["codec_name"] for s in streams if s["codec_type"] == "audio"]
        audio_args = (
            ["-c:a", "copy"] if audio_codecs == ["aac"] else ["-c:a", "aac", "-b:a", "192k"]
        )
        logger.info(f"Downscale {src_file_rel} from {width}x{height} to {out_file}")
        action_names = ["downscale"]
        if is_new_extension:
            action_names.append("convert_to_mp4")

        # if the source was written under a different name it stays around, otherwise it is replaced
        delete_source = args.overwrite and tgt_file_rel != src_file_rel
        if delete_source:
            logger.info(f"Delete {src_file_path} which was renamed to {out_file}")
            action_names.append("delete_source")
        actions[",".join(action_names)] += 1
        if not args.write:
            continue

        # encode to a temporary file so interrupted runs do not leave broken outputs behind
        tmp_file = out_file.parent / f"{out_file.stem}{TMP_SUFFIX}"
        os.makedirs(out_file.parent, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats",
            "-y",
            "-i",
            src_file_path.as_posix(),
            "-vf",
            scale_filter,
            "-c:v",
            "libx265",
            "-crf",
            str(args.crf),
            "-preset",
            args.preset,
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error",
            *audio_args,
            "-map_metadata",
            "0",
            "-movflags",
            "+faststart",
            tmp_file.as_posix(),
        ]
        logger.debug(" ".join(cmd))
        subprocess.run(cmd, check=True)

        src_stat = src_file_path.stat()
        tmp_file.replace(out_file)
        os.utime(out_file, (src_stat.st_atime, src_stat.st_mtime))
        if delete_source:
            src_file_path.unlink()
        size_before, size_after = src_stat.st_size, out_file.stat().st_size
        total_size_before += size_before
        total_size_after += size_after
        logger.info(
            f"Wrote {out_file}: {size_before / 1024**2:.1f} MB -> {size_after / 1024**2:.1f} MB"
        )

    if len(unreadable) > 0:
        logger.warning(f"Skipped {len(unreadable)} unreadable videos: {unreadable[:5]}")
    summary = {k: v for k, v in actions.most_common() if v}
    logger.info(f"Videos: {len(files2)}, actions: {summary}")
    if args.write:
        logger.info(
            f"Total: {total_size_before / 1024**2:.1f} MB -> {total_size_after / 1024**2:.1f} MB"
        )
    else:
        logger.warning("Dry run. Use -w to write changes to disk.")


if __name__ == "__main__":
    main()
