"""
Guess the rotation of videos and turn them upright.

Screenshots are taken across each video and classified by the same pretrained orientation model
the photo tool uses, see cullet-fix-rotation-images. A video is only rotated when at least
-a of its screenshots agree on the same rotation, so one misread frame cannot turn a video.
ffmpeg applies the display rotation of the container when it decodes, so only videos that a
player still shows wrong are reported. Predictions are cached per input dir.

With -w the rotated videos are written to input_dir/rotation_corrected_output keeping their
relative path, and for each a check_rotation/NAME.jpg with a screenshot before on the left and
after on the right. Videos that are upright are not copied. With -O the videos are replaced in
place instead. Either way the modification time is kept.

By default only the display rotation of the container is rewritten and the streams are copied,
which is instant, lossless and independent of the codec, and keeps the file the format it was.
With -P the rotated pixels are re-encoded instead, which every tool honors but costs time and
a generation of quality. A pixel rotation always writes hevc in mp4 whatever went in, so the
extension can change.

A container that cannot store a rotation, avi and mpeg-ts, is refused by the default path. -A
re-encodes only those and rewrites the container rotation for everything else, so a mixed
folder goes through in one pass.

Examples:
    # dry run: list which videos would be rotated
    cullet-fix-rotation-videos /path/to/videos
    # write the rotated videos and the check images to the output dir
    cullet-fix-rotation-videos /path/to/videos -w
    # rotate the pixels in place, re-encoding
    cullet-fix-rotation-videos /path/to/videos -P -O -w
"""

import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from attrs import define
from natsort import natsorted

from tqdm import tqdm
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from cullet.dedup.common import EmbeddingCache
from cullet.dedup.files import PathSpecArgs, index_files, write_json
from cullet.images import save_image
from cullet.logs import configure_logging
from cullet.paths import is_inside_dir

from cullet.rotation import (
    N_ROTATIONS,
    RESIZE_SIZE,
    OrientationClassifier,
    combine_rotation_views,
    get_default_device,
    make_check_image,
    rotate_image,
)
from cullet.video import (
    DEFAULT_CRF,
    DEFAULT_PRESET,
    TMP_STEM_SUFFIX,
    ScreenshotError,
    VideoRotation,
    execute_video_rotations,
    plan_video_rotations,
    probe_video,
    screenshot_timestamps,
    take_screenshot,
    validate_rotation_flags,
)

logger = logging.getLogger(__name__)


@define
class Args(PathSpecArgs, VerboseQuietArgs):
    input_dir: Optional[Path] = add_argument(positional=True, type=str, help="Input video dir")
    non_recursive: bool = add_argument(
        shortcut="-r", action="store_true", help="Do not search subdirectories for videos."
    )
    min_confidence: float = add_argument(
        shortcut="-c",
        type=float,
        default=0.9,
        help="A screenshot only votes for a rotation if the model is this confident.",
    )
    min_agreement: float = add_argument(
        shortcut="-a",
        type=float,
        default=0.75,
        help="Fraction of all screenshots of a video that must vote for the same rotation, "
        "otherwise the video is listed as unsure and left alone.",
    )
    screenshot_interval: float = add_argument(
        shortcut="-s", type=float, default=30.0, help="Seconds of video per screenshot."
    )
    min_screenshots: int = add_argument(
        shortcut="-n", type=int, default=5, help="Minimum screenshots per video."
    )
    max_screenshots: int = add_argument(
        shortcut="-N", type=int, default=20, help="Maximum screenshots per video."
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
        help="Re-encode the rotated pixels of every video instead of only rewriting the display "
        "rotation of the container.",
    )
    rotate_pixels_if_needed: bool = add_argument(
        shortcut="-A",
        action="store_true",
        help="Re-encode the rotated pixels only for the containers that cannot store a rotation "
        "(avi, mpeg-ts) and rewrite the container rotation for all the others, instead of "
        "refusing the whole run.",
    )
    output_dir_rel: Path = add_argument(
        shortcut="-o",
        type=str,
        help="Output directory relative to input dir.",
        default="rotation_corrected_output",
    )
    overwrite: bool = add_argument(
        shortcut="-O", action="store_true", help="Overwrite files instead of a separate output dir"
    )
    batch_size: int = add_argument(shortcut="-b", type=int, default=32, help="Model batch size")
    device: str = add_argument(type=str, default=get_default_device(), help="Torch device")
    model_file: Optional[Path] = add_argument(
        shortcut="-m", type=str, help="Model weights, default: downloaded to cache dir."
    )
    reset_cache: bool = add_argument(action="store_true", help="Recompute all predictions.")
    report_file: Optional[Path] = add_argument(
        type=str, help="Write the guessed rotation of every video to this json file."
    )
    write: bool = add_argument(shortcut="-w", action="store_true", help="Write changes to disk")


# containers that can store a rotation, so the default path keeps them as they are, plus the
# two that cannot and therefore need -A or -P to be rotated at all
ENDINGS = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".mts"]
CHECK_DIR_NAME = "check_rotation"
CHECK_QUALITY = 85


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")
    assert 0 < args.min_screenshots <= args.max_screenshots, f"{args}"
    assert 0 < args.min_agreement <= 1, f"Agreement must be in (0, 1]: {args.min_agreement}"
    validate_rotation_flags(args.rotate_pixels, args.rotate_pixels_if_needed)
    input_dir = Path(args.input_dir)
    output_dir = None if args.overwrite else input_dir / args.output_dir_rel

    rel_files, entries, broken_files = compute_predictions(input_dir, args, output_dir)

    # decide per video, a rotation needs enough screenshots to agree on it
    to_rotate = {}
    unsure = []
    report = []
    actions = Counter()
    actions["unreadable"] += len(broken_files)
    for rel_file in rel_files:
        entry = entries[rel_file]
        probs = combine_rotation_views(entry["logp"])
        rotation, agreement, n_votes = vote_on_rotation(probs, args.min_confidence)
        report.append(
            {
                "file": rel_file,
                "rotation_cw": rotation * 90,
                "agreement": agreement,
                "n_screenshots": len(probs),
            }
        )
        described = describe(entry, len(probs), agreement, n_votes)
        if agreement < args.min_agreement:
            logger.warning(f"UNSURE {described} {rel_file}")
            unsure.append(rel_file)
            actions["unsure"] += 1
            continue
        if rotation == 0:
            logger.debug(f"OK     {described} {rel_file}")
            actions["upright"] += 1
            continue
        logger.info(f"ROTATE {described} {rotation * 90:3d}° cw {rel_file}")
        to_rotate[rel_file] = rotation

    if args.report_file is not None:
        write_json(report, Path(args.report_file))
    if len(broken_files) > 0:
        logger.warning(f"Skipped {len(broken_files)} unreadable files: {list(broken_files)[:5]}")
    logger.info(
        f"Videos: {len(rel_files)}, upright: {actions['upright']}, "
        f"to rotate: {len(to_rotate)}, unsure: {len(unsure)}"
    )
    requests = [
        (
            input_dir / rel_file,
            (input_dir if output_dir is None else output_dir) / rel_file,
            rotation * 90,
        )
        for rel_file, rotation in natsorted(to_rotate.items())
    ]
    rotations = plan_video_rotations(
        requests,
        rotate_pixels=args.rotate_pixels,
        rotate_pixels_if_needed=args.rotate_pixels_if_needed,
        delete_source=args.overwrite,
    )

    def check_rotation(rotation: VideoRotation) -> None:
        assert output_dir is not None
        rel_file = rotation.src_file.relative_to(input_dir)
        write_check_image(
            rotation.src_file,
            output_dir / CHECK_DIR_NAME / f"{rel_file}.jpg",
            rotation.rotation_cw // 90,
            args,
        )

    execute_video_rotations(
        rotations,
        write=args.write,
        crf=args.crf,
        preset=args.preset,
        actions=actions,
        before_rotate=check_rotation if output_dir is not None else None,
    )


def vote_on_rotation(probs: torch.Tensor, min_confidence: float) -> tuple[int, float, int]:
    """
    Let every confident screenshot vote for its rotation and return the winner.

    Args:
        probs: (n_screenshots, 4) rotation probabilities
        min_confidence: a screenshot only votes if its best rotation reaches this

    Returns:
        the winning rotation in steps counter-clockwise, the fraction of all screenshots that
        voted for it, and the number of screenshots that voted at all
    """
    confidences, rotations = probs.max(dim=-1)
    votes = Counter(
        int(rotation)
        for rotation, confidence in zip(rotations.tolist(), confidences.tolist())
        if confidence >= min_confidence
    )
    if len(votes) == 0:
        return 0, 0.0, 0
    rotation, n_winner = votes.most_common(1)[0]
    return rotation, n_winner / len(probs), sum(votes.values())


def describe(entry: dict, n_screenshots: int, agreement: float, n_votes: int) -> str:
    width, height = int(entry["width"]), int(entry["height"])
    return (
        f"{width:5d}x{height:<5d} {float(entry['duration']):7.1f}s "
        f"agree={agreement:.2f} votes={n_votes}/{n_screenshots}"
    )


def write_check_image(src_file: Path, check_file: Path, rotation: int, args: Args) -> None:
    """Save a screenshot from the middle of the video, before on the left and after on the right."""
    info = probe_video(src_file)
    try:
        frame = take_screenshot(src_file, info.duration / 2)
    except (subprocess.CalledProcessError, ScreenshotError) as e:
        logger.warning(f"No check image for {src_file}: {e}")
        return
    rotated = rotate_image(frame, (N_ROTATIONS - rotation) % N_ROTATIONS)
    save_image(make_check_image(frame, rotated), check_file, quality=CHECK_QUALITY)


def compute_predictions(
    input_dir: Path, args: Args, output_dir: Path | None
) -> tuple[list[str], dict[str, dict], dict[str, str]]:
    """
    Classify screenshots of all videos in the directory, reusing cached predictions.

    Returns:
        relative paths of the readable videos in name order, their cache entries with keys
        logp (n_screenshots, 4, 4), width, height, duration, rotation, and the unreadable
        files mapped to their error
    """
    src_index = index_files(input_dir, not args.non_recursive, args)
    logger.info(f"Found {len(src_index)} files in {input_dir}")
    rel_files = natsorted(
        f
        for f in src_index.keys()
        if any(f.lower().endswith(a) for a in ENDINGS)
        and not is_inside_dir(f, input_dir, output_dir)
        and not (Path(f).name.startswith(".") and Path(f).stem.endswith(TMP_STEM_SUFFIX))
    )
    logger.info(f"From those detected {len(rel_files)} videos.")
    if len(rel_files) == 0:
        raise ValueError(
            f"No videos found in {input_dir} ({len(src_index)} files indexed, "
            f"video endings: {ENDINGS}, subdirectories searched: {not args.non_recursive})"
        )
    cache = EmbeddingCache("rotation_videos", input_dir, reset=args.reset_cache)
    entries = {}
    todo = []
    for rel_file in rel_files:
        props = src_index[rel_file]
        entry = cache.get(rel_file, props.st_size, props.st_mtime)
        if entry is None or entry["screenshot_params"] != screenshot_params(args):
            todo.append(rel_file)
        else:
            entries[rel_file] = entry
    logger.info(f"{len(entries)} videos cached, {len(todo)} to process.")
    broken_files = {}
    if len(todo) > 0:
        classifier = OrientationClassifier(args.model_file, args.device, args.batch_size)
        pbar = tqdm(total=len(todo), desc="Classifying")
        for rel_file in todo:
            pbar.update()
            video_file = input_dir / rel_file
            try:
                info = probe_video(video_file)
                timestamps = screenshot_timestamps(
                    info.duration,
                    args.screenshot_interval,
                    args.min_screenshots,
                    args.max_screenshots,
                )
                screenshots = [take_screenshot(video_file, t, size=RESIZE_SIZE) for t in timestamps]
            # any failure reading one video, e.g. no video stream or a truncated file, skips it
            # instead of ending the run
            except Exception as e:
                error = (
                    e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else repr(e)
                )
                logger.error(f"Skipping {rel_file}: {error}")
                broken_files[rel_file] = error
                continue
            tensors = torch.stack([classifier.transform(s) for s in screenshots])
            props = src_index[rel_file]
            entries[rel_file] = cache.put(
                rel_file,
                props.st_size,
                props.st_mtime,
                logp=classifier.classify_rotations(tensors),
                width=info.display_size[0],
                height=info.display_size[1],
                duration=info.duration,
                rotation=info.rotation,
                screenshot_params=screenshot_params(args),
            )
        pbar.close()
        cache.save(keep_rel_files=rel_files)
    if len(broken_files) > 0:
        logger.warning(f"Skipped {len(broken_files)} unreadable videos.")
    rel_files = [f for f in rel_files if f in entries]
    if len(rel_files) == 0:
        raise ValueError(f"None of the videos in {input_dir} could be read: {broken_files}")
    return rel_files, entries, broken_files


def screenshot_params(args: Args) -> tuple[str, float, int, int]:
    # v2 explicitly selects the first video stream, matching probing and rotation.
    return "v2", args.screenshot_interval, args.min_screenshots, args.max_screenshots


if __name__ == "__main__":
    main()
