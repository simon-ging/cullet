"""
Probe videos, decode single frames as screenshots and rotate videos, all with ffmpeg.

ffmpeg applies the display rotation of a video when it decodes, so screenshots come out the
way a player shows them, the same way the exif orientation is applied when loading an image.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from attrs import define
from PIL import Image

logger = logging.getLogger(__name__)

# when seeking past the last frame, decode from this many seconds earlier and use the last frame
SCREENSHOT_FALLBACK_SECONDS = 2.0
# unfinished outputs of interrupted runs
TMP_STEM_SUFFIX = ".tmp"
# rotating re-encodes an already compressed video, so the target is transparency rather than
# a small file. crf 18 measured about 40 dB psnr against the source, crf 23 about 38 dB.
DEFAULT_CRF = 18
DEFAULT_PRESET = "slow"
# a pixel rotation always writes hevc in mp4, whatever went in. one target means one encoder
# and one quality flag instead of a mapping from every codec that can appear
ROTATE_TARGET_ENCODER = "libx265"
ROTATE_TARGET_EXTENSION = ".mp4"
# containers that want the hvc1 tag so apple software plays the hevc stream
HVC1_EXTENSIONS = (".mp4", ".mov", ".m4v")
# ffmpeg filters that rotate the pixels clockwise, transpose=1 is 90 degrees clockwise
ROTATE_FILTERS = {90: "transpose=1", 180: "transpose=1,transpose=1", 270: "transpose=2"}
# containers that can store a display rotation. avi has no field for it and mpeg-ts none that
# ffmpeg writes, so a rotation would be dropped without any error, see can_store_rotation()
ROTATION_METADATA_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".3gp")


class ScreenshotError(RuntimeError):
    """Raised when ffmpeg decodes no frame for a video, e.g. a video track without frames."""


@define
class VideoInfo:
    width: int  # as stored in the stream, before the display rotation is applied
    height: int
    duration: float  # seconds
    rotation: int  # display rotation in degrees counter-clockwise, 0 when there is none
    codec: str = ""  # as ffmpeg names it, e.g. h264
    # (stream index, codec tag) of streams whose codec ffmpeg does not know, so it cannot copy
    # them, e.g. the mebx timed metadata tracks of iphone videos. rotating drops them
    uncopyable_streams: tuple[tuple[int, str], ...] = ()

    @property
    def display_size(self) -> tuple[int, int]:
        """Width and height as a player shows them, so with the display rotation applied."""
        if self.rotation % 180 == 0:
            return self.width, self.height
        return self.height, self.width


def probe_video(video_file: Path) -> VideoInfo:
    """
    Read size, duration and display rotation of the first video stream.

    The container duration can be longer than the video track, e.g. when the audio track is
    longer, so the video track duration is used when the container reports one. All streams
    are read to find the ones ffmpeg cannot copy.
    """
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_tag_string,width,height,duration,codec_name"
            ":stream_side_data=rotation:format=duration",
            "-of",
            "json",
            video_file.as_posix(),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    info = json.loads(output)
    video_streams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    assert len(video_streams) > 0, f"No video stream in {video_file}: {info}"
    stream = video_streams[0]
    uncopyable_streams = tuple(
        (s["index"], s.get("codec_tag_string", ""))
        for s in info["streams"]
        if s.get("codec_name", "unknown") == "unknown"
    )
    duration = float(info["format"]["duration"])
    if "duration" in stream:
        duration = min(duration, float(stream["duration"]))
    rotation = 0
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(side_data["rotation"])
    return VideoInfo(
        stream["width"],
        stream["height"],
        duration,
        rotation,
        stream.get("codec_name", ""),
        uncopyable_streams,
    )


def screenshot_timestamps(
    duration: float, interval: float, min_screenshots: int, max_screenshots: int
) -> list[float]:
    """
    Screenshots are taken on a fixed grid from the start of the video, so a copy that was
    trimmed at the end still has its screenshots at the same times as the original. Videos too
    long for the grid use a multiple of the interval, videos too short spread the minimum number
    of screenshots evenly.
    """
    grid_factor = max(1, math.ceil(duration / (interval * max_screenshots)))
    grid_interval = interval * grid_factor
    n_screenshots = math.floor(duration / grid_interval)
    if n_screenshots < min_screenshots:
        return [duration * (i + 0.5) / min_screenshots for i in range(min_screenshots)]
    return [grid_interval * (i + 0.5) for i in range(n_screenshots)]


def take_screenshot(video_file: Path, timestamp: float, size: int | None = None) -> Image.Image:
    """
    Decode one frame at the timestamp.

    Seeking returns the first frame at or after the timestamp, which is nothing when the
    timestamp is past the last frame of a short or low fps video. Then the frames from a bit
    earlier are decoded and the last one is used.

    Args:
        video_file: input video
        timestamp: seconds from the start
        size: scale the frame to this many pixels on both sides, skewing the aspect ratio the
            same way the models do. None keeps the frame as decoded.

    Returns:
        the frame as an rgb image
    """
    frames = decode_frames(video_file, timestamp, size=size, max_frames=1)
    if len(frames) == 0:
        logger.debug(f"No frame at {timestamp:.3f}s of {video_file}, using the last one before")
        frames = decode_frames(
            video_file, max(0.0, timestamp - SCREENSHOT_FALLBACK_SECONDS), size=size
        )
    if len(frames) == 0:
        raise ScreenshotError(f"No decodable video frame at {timestamp:.3f}s of {video_file}")
    return Image.fromarray(frames[-1])


def decode_frames(
    video_file: Path, start: float, size: int | None = None, max_frames: int | None = None
) -> list[np.ndarray]:
    """Decode frames from the start time until the end or max_frames, as (H, W, 3) rgb arrays."""
    max_frames_args = [] if max_frames is None else ["-frames:v", str(max_frames)]
    scale_args = [] if size is None else ["-vf", f"scale={size}:{size}"]
    output = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            video_file.as_posix(),
            "-map",
            "0:v:0",
            *max_frames_args,
            *scale_args,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if size is not None:
        n_bytes_per_frame = size * size * 3
        assert len(output) % n_bytes_per_frame == 0, f"Got {len(output)} bytes from {video_file}"
        return list(np.frombuffer(output, dtype=np.uint8).reshape(-1, size, size, 3))
    # without scaling the frame size has to be probed to reshape the raw bytes
    info = probe_video(video_file)
    width, height = info.display_size
    n_bytes_per_frame = width * height * 3
    assert len(output) % n_bytes_per_frame == 0, (
        f"Got {len(output)} bytes from {video_file}, expected a multiple of "
        f"{n_bytes_per_frame} for {width}x{height}"
    )
    return list(np.frombuffer(output, dtype=np.uint8).reshape(-1, height, width, 3))


def can_store_rotation(video_file: Path) -> bool:
    """Whether the container of the file can hold a display rotation."""
    return video_file.suffix.lower() in ROTATION_METADATA_EXTENSIONS


def rotate_target_file(out_file: Path, rotate_pixels: bool) -> Path:
    """
    Where a rotation actually writes.

    A pixel rotation always produces hevc in mp4, so the extension changes for any other
    container. A container rotation copies the streams and keeps the file as it is.
    """
    if not rotate_pixels:
        return out_file
    return out_file.with_suffix(ROTATE_TARGET_EXTENSION)


@define(frozen=True)
class VideoRotation:
    src_file: Path
    out_file: Path
    rotation_cw: int
    info: VideoInfo
    rotate_pixels: bool
    delete_source: bool


def validate_rotation_flags(rotate_pixels: bool, rotate_pixels_if_needed: bool) -> None:
    if rotate_pixels and rotate_pixels_if_needed:
        raise ValueError(
            "Specify either -P to re-encode every video or -A to re-encode only the containers "
            "that need it, not both."
        )


def _check_rotation_target(src_file: Path, out_file: Path) -> None:
    if not src_file.is_file():
        raise FileNotFoundError(f"Not a file: {src_file}")
    if (out_file.exists() or out_file.is_symlink()) and (
        not out_file.exists() or not out_file.samefile(src_file)
    ):
        raise FileExistsError(
            f"Refusing to rotate {src_file} over the existing {out_file}, which is a different "
            f"file. Remove it or pick another output directory."
        )


def plan_video_rotations(
    requests: list[tuple[Path, Path, int]],
    *,
    rotate_pixels: bool = False,
    rotate_pixels_if_needed: bool = False,
    delete_source: bool = False,
) -> list[VideoRotation]:
    """Resolve (source, output base, clockwise angle) requests and validate the entire batch.

    No output is created here. Both CLIs use the same container policy, target collision
    checks and probing before any video or check image is written.
    """
    validate_rotation_flags(rotate_pixels, rotate_pixels_if_needed)
    pending = []
    targets = {}
    unsupported = []
    for src_file, output_base, rotation_cw in requests:
        if rotation_cw not in ROTATE_FILTERS:
            raise ValueError(
                f"Cannot rotate by {rotation_cw} degrees, only {sorted(ROTATE_FILTERS)} are possible"
            )
        pixels = rotate_pixels or (rotate_pixels_if_needed and not can_store_rotation(src_file))
        out_file = rotate_target_file(output_base, pixels)
        if not pixels and not can_store_rotation(out_file):
            unsupported.append(src_file)
        target = out_file.resolve()
        if target in targets:
            raise ValueError(
                f"{src_file} and {targets[target]} would both be rotated into {out_file}. "
                f"Rename one of them first."
            )
        targets[target] = src_file
        pending.append((src_file, out_file, rotation_cw, pixels))
    if unsupported:
        raise ValueError(
            f"These videos cannot store a display rotation: {unsupported}. Pass -A to re-encode "
            f"only those, -P to re-encode everything, or exclude them. Containers that can "
            f"store a rotation: {', '.join(ROTATION_METADATA_EXTENSIONS)}"
        )
    for src_file, out_file, _, _ in pending:
        _check_rotation_target(src_file, out_file)
    rotations = []
    for src_file, out_file, angle, pixels in pending:
        # a video ffprobe cannot read is reported and left out, so it does not stop the batch
        try:
            info = probe_video(src_file)
        except (subprocess.CalledProcessError, AssertionError) as e:
            error = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
            logger.error(f"Skipping unreadable {src_file}: {error}")
            continue
        rotations.append(VideoRotation(src_file, out_file, angle, info, pixels, delete_source))
    return rotations


def execute_video_rotations(
    rotations: list[VideoRotation],
    *,
    write: bool = False,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    actions: dict[str, int] | None = None,
    before_rotate: Callable[[VideoRotation], None] | None = None,
) -> None:
    """Report a validated batch, then optionally write it using the shared rotation tooling."""
    counts = Counter(actions or {})
    for rotation in rotations:
        pixels = rotation.rotate_pixels
        action = "rotate,reencode" if pixels else "rotate,container"
        method = "re-encode to hevc mp4" if pixels else "container"
        width, height = rotation.info.display_size
        logger.info(
            f"Rotate {rotation.rotation_cw}° cw ({method}) {width}x{height} "
            f"{rotation.info.duration:.1f}s: {rotation.src_file} -> {rotation.out_file}"
        )
        if rotation.info.uncopyable_streams:
            action += ",drop_streams"
            tags = [tag for _, tag in rotation.info.uncopyable_streams]
            logger.warning(
                f"Dropping {len(tags)} streams that ffmpeg cannot copy, codec tags {tags}: "
                f"{rotation.src_file}"
            )
        counts[action] += 1
    summary = {k: v for k, v in counts.most_common() if v}
    logger.info(f"Actions: {summary}")
    if not write:
        logger.warning("Dry run. Use -w to write the rotated videos.")
        return
    # Recheck the whole batch in case a target appeared since planning.
    for rotation in rotations:
        _check_rotation_target(rotation.src_file, rotation.out_file)
    failed = {}
    for rotation in rotations:
        # one video that ffmpeg fails on is reported and the batch continues. rotate_video only
        # replaces the target once ffmpeg succeeded, so a failure leaves the source untouched
        try:
            if before_rotate is not None:
                before_rotate(rotation)
            rotate_video(
                rotation.src_file,
                rotation.out_file,
                rotation.rotation_cw,
                rotation.info,
                crf=crf,
                preset=preset,
                rotate_pixels=rotation.rotate_pixels,
                delete_source=rotation.delete_source,
            )
        except Exception as e:
            error = e.stderr.strip() if getattr(e, "stderr", None) else repr(e)
            logger.error(f"Failed to rotate {rotation.src_file}: {error}")
            failed[rotation.src_file] = error
    done = [rotation for rotation in rotations if rotation.src_file not in failed]
    n_reencoded = sum(rotation.rotate_pixels for rotation in done)
    logger.info(
        f"Rotated {len(done)} videos, {n_reencoded} by re-encoding and "
        f"{len(done) - n_reencoded} by rewriting the container rotation."
    )
    if failed:
        logger.error(f"Failed to rotate {len(failed)} videos: {[str(f) for f in failed]}")


def build_rotate_command(
    src_file: Path,
    out_file: Path,
    rotation_cw: int,
    info: VideoInfo,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    rotate_pixels: bool = False,
) -> list[str]:
    """
    Build the ffmpeg command that rotates the first video stream clockwise.

    By default only the display rotation of the container is rewritten and the streams are
    copied, which is instant, lossless and does not care about the codec. rotate_pixels decodes
    and re-encodes to hevc in mp4 instead, which every tool honors but costs time and a
    generation of quality. Every stream is mapped explicitly: other video streams are copied,
    and pixel rotation converts all audio tracks to AAC. An incompatible extra stream causes
    ffmpeg to fail instead of being silently discarded. The exception are streams whose codec
    ffmpeg does not know at all, see VideoInfo.uncopyable_streams, which no muxer can write and
    are dropped with a warning by the callers.

    Args:
        src_file: input video
        out_file: where ffmpeg writes, see rotate_target_file for the expected extension
        rotation_cw: 90, 180 or 270 degrees clockwise
        info: probe result of the input, for its current display rotation
        crf: x265 quality for rotate_pixels, lower is better and bigger
        preset: x265 preset for rotate_pixels
        rotate_pixels: re-encode the rotated pixels instead of rewriting the container

    Returns:
        the ffmpeg command

    Raises:
        ValueError: for a rotation angle other than 90, 180 or 270, for a container that cannot
            store a rotation while rotate_pixels is off, or for a pixel rotation into anything
            but mp4
    """
    if rotation_cw not in ROTATE_FILTERS:
        raise ValueError(
            f"Cannot rotate by {rotation_cw} degrees, only {sorted(ROTATE_FILTERS)} are possible"
        )
    map_args = ["-map", "0"]
    for index, _ in info.uncopyable_streams:
        map_args += ["-map", f"-0:{index}"]
    if not rotate_pixels:
        if not can_store_rotation(out_file):
            raise ValueError(
                f"A {out_file.suffix} container cannot store a display rotation, so "
                f"{out_file.name} would come out unrotated without any error from ffmpeg. "
                f"Pass -A to re-encode only the containers that need it, or -P to re-encode "
                f"everything. Containers that can store a rotation: "
                f"{', '.join(ROTATION_METADATA_EXTENSIONS)}"
            )
        # the display rotation counts counter-clockwise and replaces the one of the input
        new_rotation = (info.rotation - rotation_cw) % 360
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation:v:0",
            str(new_rotation),
            "-i",
            src_file.as_posix(),
            *map_args,
            "-copy_unknown",
            "-c",
            "copy",
            "-map_metadata",
            "0",
            out_file.as_posix(),
        ]
    if out_file.suffix.lower() != ROTATE_TARGET_EXTENSION:
        raise ValueError(
            f"A pixel rotation writes {ROTATE_TARGET_ENCODER} which needs a "
            f"{ROTATE_TARGET_EXTENSION} container, not {out_file.suffix}. Use "
            f"rotate_target_file() to pick the output path."
        )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src_file.as_posix(),
        *map_args,
        "-copy_unknown",
        "-c",
        "copy",
        "-filter:v:0",
        ROTATE_FILTERS[rotation_cw],
        "-c:v:0",
        ROTATE_TARGET_ENCODER,
        "-crf",
        str(crf),
        "-preset",
        preset,
        # the hvc1 tag makes apple software play the hevc stream
        "-tag:v:0",
        "hvc1",
        "-x265-params",
        "log-level=error",
        # aac plays in mp4, anything else the source may carry is converted
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        out_file.as_posix(),
    ]


def rotate_video(
    src_file: Path,
    out_file: Path,
    rotation_cw: int,
    info: VideoInfo,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    rotate_pixels: bool = False,
    delete_source: bool = False,
) -> None:
    """
    Rotate a video clockwise, see build_rotate_command for the rest of the arguments.

    ffmpeg writes to a temporary file next to the output, which is then renamed over it. An
    interrupted run therefore leaves the source untouched and at worst a leftover temporary
    file, never a half written output. out_file may be the same path as src_file.

    delete_source removes the input afterwards, for replacing a video whose container changed,
    e.g. a webm rotated into an mp4. It only runs once the output is complete and is skipped
    when both paths are the same file, so an interruption can leave a copy but never a gap.
    """
    _check_rotation_target(src_file, out_file)
    cmd = build_rotate_command(src_file, out_file, rotation_cw, info, crf, preset, rotate_pixels)
    src_stat = src_file.stat()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a unique name atomically. A timestamp alone can collide, and NAME.tmp.mp4
    # might be another input video. Keep the real extension so ffmpeg chooses the container.
    fd, tmp_name = tempfile.mkstemp(
        dir=out_file.parent,
        prefix=f".{out_file.stem}.",
        suffix=f"{TMP_STEM_SUFFIX}{out_file.suffix}",
    )
    os.close(fd)
    tmp_file = Path(tmp_name)
    cmd[-1] = tmp_file.as_posix()
    logger.debug(" ".join(cmd))
    # remove the reserved temporary file when ffmpeg fails or is interrupted, otherwise every
    # failed video leaves an empty hidden file in the output directory. the error is re-raised
    try:
        subprocess.run(cmd, check=True)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise
    _check_rotation_target(src_file, out_file)
    tmp_file.replace(out_file)
    os.utime(out_file, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
    if delete_source and not out_file.samefile(src_file):
        logger.debug(f"Removing {src_file}, rotated into {out_file}")
        src_file.unlink()
