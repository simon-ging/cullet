"""
Find duplicate videos regardless of their size, using ffmpeg screenshots and the SSCD model.

Two videos are compared only if their durations are within a few seconds or percent (-d/-D).
A screenshot is taken every 30 seconds from the start (-s), at least 5 and at most 20 per video
(-n/-N). Each screenshot is embedded with SSCD and matched to the most similar screenshot of the
other video. The videos are duplicates if the average of these best matches is at least -t.

In each group of duplicates one video is kept: the one with at least 5% more pixels, else the
one at least 1% longer, else the one with at least 5% bigger file size, else the first in name
order. The other videos are moved to the trash, or to a quarantine dir with -Q, or deleted
permanently with --unlink. Screenshot embeddings are cached per input dir, so rerunning only
processes new videos. Needs the ffmpeg and ffprobe binaries.

Example: cullet-dedup-videos /path/to/videos -Q /path/to/quarantine -w
"""

import json
import logging
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from attrs import define
from natsort import natsorted
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from cullet.dedup.files import PathSpecArgs, index_files, remove_duplicates
from cullet.extras import require_dedup_extra
from cullet.logs import configure_logging
from cullet.paths import is_inside_dir

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@define
class Args(PathSpecArgs, VerboseQuietArgs):
    input_dir: Optional[Path] = add_argument(positional=True, type=str, help="Input video dir")
    non_recursive: bool = add_argument(
        shortcut="-r", action="store_true", help="Do not search subdirectories for videos."
    )
    duration_tolerance: float = add_argument(
        shortcut="-d",
        type=float,
        default=3.0,
        help="Only compare videos whose durations differ at most this many seconds, "
        "or at most the fraction -D of the longer video, whichever is larger.",
    )
    duration_tolerance_rel: float = add_argument(
        shortcut="-D",
        type=float,
        default=0.1,
        help="Relative duration tolerance, see -d. Copies are often trimmed a bit at the end.",
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
    threshold: float = add_argument(
        shortcut="-t",
        type=float,
        default=0.9,
        help="Average best-match cosine similarity of screenshots to count as duplicates. "
        "Re-encoded copies score above 0.93, unrelated videos below 0.5.",
    )
    batch_size: int = add_argument(shortcut="-b", type=int, default=32, help="Model batch size")
    device: Optional[str] = add_argument(
        type=str, help="Torch device, default: cuda if available, else cpu"
    )
    model_file: Optional[Path] = add_argument(
        shortcut="-m", type=str, help="SSCD torchscript file, default: downloaded to cache dir."
    )
    reset_cache: bool = add_argument(action="store_true", help="Recompute all embeddings.")
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


ENDINGS = [".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp", ".mts"]
# pixels, duration or file size only decide which duplicate to keep if they differ by more than
# this. copies of a video differ by a frame or so, a trimmed copy is much shorter.
KEEP_TOLERANCE = 0.05
KEEP_TOLERANCE_DURATION = 0.01


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logging(args)
    logger.info(f"{args}")
    require_dedup_extra()
    # torch is only loaded behind the check, so a missing extra gives a message that helps
    from tqdm import tqdm

    from cullet.dedup.common import EmbeddingCache, group_pairs, pick_best_member
    from cullet.dedup.sscd import SSCD_INPUT_SIZE, SscdEmbedder
    from cullet.dedup.video import (
        ScreenshotError,
        probe_video,
        screenshot_timestamps,
        take_screenshot,
    )

    assert 0 < args.min_screenshots <= args.max_screenshots, f"{args}"
    input_dir = Path(args.input_dir)
    quarantine_dir = None if args.quarantine_dir is None else Path(args.quarantine_dir)

    src_index = index_files(input_dir, not args.non_recursive, args)
    logger.info(f"Found {len(src_index)} files.")
    rel_files = natsorted(
        f
        for f in src_index.keys()
        if any(f.lower().endswith(a) for a in ENDINGS)
        and not is_inside_dir(f, input_dir, quarantine_dir)
    )
    logger.info(f"From those detected {len(rel_files)} videos.")
    if len(rel_files) == 0:
        raise ValueError(
            f"No videos found in {input_dir} ({len(src_index)} files indexed, "
            f"video endings: {ENDINGS}, subdirectories searched: {not args.non_recursive})"
        )

    # probe and screenshot the videos that are not in the cache yet
    cache = EmbeddingCache("videos", input_dir, reset=args.reset_cache)
    entries = {}
    todo = []
    for rel_file in rel_files:
        props = src_index[rel_file]
        entry = cache.get(rel_file, props.st_size, props.st_mtime)
        if entry is None or entry["screenshot_params"] != _screenshot_params(args):
            todo.append(rel_file)
        else:
            entries[rel_file] = entry
    logger.info(f"{len(entries)} videos cached, {len(todo)} to process.")
    broken_files = {}
    if len(todo) > 0:
        embedder = SscdEmbedder(args.model_file, args.device, args.batch_size)
        pbar = tqdm(total=len(todo), desc="Screenshots")
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
                screenshots = [
                    take_screenshot(video_file, t, size=SSCD_INPUT_SIZE) for t in timestamps
                ]
            except (subprocess.CalledProcessError, ScreenshotError) as e:
                error = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
                logger.error(f"Skipping {rel_file}: {error}")
                broken_files[rel_file] = error
                continue
            props = src_index[rel_file]
            entries[rel_file] = cache.put(
                rel_file,
                props.st_size,
                props.st_mtime,
                embs=embedder.embed_pil_images(screenshots),
                width=info.width,
                height=info.height,
                duration=info.duration,
                screenshot_params=_screenshot_params(args),
            )
        pbar.close()
        cache.save(keep_rel_files=rel_files)
    rel_files = [f for f in rel_files if f in entries]
    if len(rel_files) == 0:
        raise ValueError(f"None of the videos in {input_dir} could be read: {broken_files}")

    # compare videos with similar duration
    by_duration = sorted(range(len(rel_files)), key=lambda i: entries[rel_files[i]]["duration"])
    pairs = []
    for pos_a, a in enumerate(by_duration):
        duration_a = entries[rel_files[a]]["duration"]
        for b in by_duration[pos_a + 1 :]:
            duration_b = entries[rel_files[b]]["duration"]
            tolerance = max(args.duration_tolerance, args.duration_tolerance_rel * duration_b)
            if duration_b - duration_a > tolerance:
                break
            sim = screenshot_similarity(
                entries[rel_files[a]]["embs"], entries[rel_files[b]]["embs"]
            )
            logger.debug(f"sim={sim:.3f} {rel_files[a]} vs {rel_files[b]}")
            if sim >= args.threshold:
                pairs.append((min(a, b), max(a, b), sim))
    groups = group_pairs(len(rel_files), pairs)
    logger.info(f"Found {len(pairs)} similar pairs forming {len(groups)} duplicate groups.")

    # inside each group keep the biggest video
    report = []
    to_remove = []
    for members in groups:
        keep = pick_best_member(
            members,
            lambda i: [
                (_n_pixels(entries[rel_files[i]]), KEEP_TOLERANCE),
                (entries[rel_files[i]]["duration"], KEEP_TOLERANCE_DURATION),
                (entries[rel_files[i]]["size"], KEEP_TOLERANCE),
            ],
        )
        logger.info(f"KEEP   {_describe(entries[rel_files[keep]])} {rel_files[keep]}")
        group_report = {"keep": rel_files[keep], "remove": []}
        for i in members:
            if i == keep:
                continue
            entry = entries[rel_files[i]]
            # similarity to the closest other member, that is what put the video into the group
            sim = max(
                screenshot_similarity(entry["embs"], entries[rel_files[j]]["embs"])
                for j in members
                if j != i
            )
            logger.info(f"  DEL  {_describe(entry)} sim={sim:.3f} {rel_files[i]}")
            group_report["remove"].append({"file": rel_files[i], "sim": sim})
            to_remove.append(rel_files[i])
        report.append(group_report)
    if args.report_file is not None:
        report_file = Path(args.report_file)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if len(broken_files) > 0:
        logger.warning(f"Skipped {len(broken_files)} unreadable files: {list(broken_files)[:5]}")
    actions = Counter()
    actions["unique"] += len(rel_files) - sum(len(members) for members in groups)
    actions["duplicate_keep"] += len(groups)
    actions["duplicate_remove"] += len(to_remove)
    actions["unreadable"] += len(broken_files)
    summary = {k: v for k, v in actions.most_common() if v}
    logger.info(
        f"Videos: {len(rel_files)}, duplicate groups: {len(groups)}, " f"actions: {summary}"
    )
    remove_duplicates(
        input_dir, to_remove, quarantine_dir, args.write, args.delete_only, args.unlink
    )


def _screenshot_params(args: Args) -> tuple[str, float, int, int]:
    # the version invalidates cache entries when the timestamp logic changes
    return "v2", args.screenshot_interval, args.min_screenshots, args.max_screenshots


def screenshot_similarity(embs_a: "torch.Tensor", embs_b: "torch.Tensor") -> float:
    """
    Match every screenshot to its most similar screenshot of the other video and average, in
    both directions. The lower direction counts, so a video does not match a longer one just
    because all of its screenshots appear somewhere in the other.
    """
    sim = embs_a @ embs_b.T
    return float(min(sim.max(dim=1).values.mean(), sim.max(dim=0).values.mean()))


def _n_pixels(entry: dict) -> int:
    return entry["width"] * entry["height"]


def _describe(entry: dict) -> str:
    return (
        f"{entry['width']:5d}x{entry['height']:<5d} {entry['duration']:7.1f}s "
        f"{entry['size'] / 1024**2:7.1f}MB"
    )


if __name__ == "__main__":
    main()
