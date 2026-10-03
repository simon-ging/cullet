"""
Probe videos and decode single frames as screenshots with ffmpeg.

ffmpeg applies the display rotation of a video when it decodes, so screenshots come out the
way a player shows them, the same way the exif orientation is applied when loading an image.
"""

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
from attrs import define
from PIL import Image

logger = logging.getLogger(__name__)
# when seeking past the last frame, decode from this many seconds earlier and use the last frame
SCREENSHOT_FALLBACK_SECONDS = 2.0


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
