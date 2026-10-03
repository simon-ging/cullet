import shutil
import subprocess

import pytest

pytest.importorskip("numpy")  # part of the dedup extra

from cullet.dedup.video import (  # noqa: E402
    VideoInfo,
    probe_video,
    screenshot_timestamps,
    take_screenshot,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)


@pytest.mark.parametrize(
    "rotation,expected",
    [(0, (1920, 1080)), (180, (1920, 1080)), (90, (1080, 1920)), (-90, (1080, 1920))],
)
def test_video_info_display_size(rotation, expected):
    assert VideoInfo(1920, 1080, 10.0, rotation).display_size == expected


def test_screenshot_timestamps_spreads_short_videos():
    stamps = screenshot_timestamps(10.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    assert stamps == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_screenshot_timestamps_uses_a_fixed_grid():
    # a copy trimmed at the end keeps its screenshots at the same times as the original
    full = screenshot_timestamps(300.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    trimmed = screenshot_timestamps(280.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    assert trimmed == full[: len(trimmed)] and len(trimmed) == 9


def test_screenshot_timestamps_respects_the_maximum():
    stamps = screenshot_timestamps(10000.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    assert len(stamps) <= 20 and stamps[-1] < 10000.0


@needs_ffmpeg
def test_probe_and_screenshot(tmp_path):
    video_file = tmp_path / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=5"]
        + ["-pix_fmt", "yuv420p", video_file.as_posix()],
        check=True,
    )
    info = probe_video(video_file)
    assert (info.width, info.height, info.rotation) == (64, 48, 0)
    assert abs(info.duration - 2.0) < 0.3
    assert take_screenshot(video_file, 1.0).size == (64, 48)
    assert take_screenshot(video_file, 1.0, size=32).size == (32, 32)
    # past the last frame the last one before is used
    assert take_screenshot(video_file, 1.99, size=32).size == (32, 32)
