import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("numpy")  # part of the full extra

from cullet import video as videotools  # noqa: E402

from cullet.video import (  # noqa: E402
    DEFAULT_CRF,
    ROTATE_FILTERS,
    ROTATE_TARGET_ENCODER,
    VideoInfo,
    build_rotate_command,
    execute_video_rotations,
    plan_video_rotations,
    probe_video,
    rotate_target_file,
    rotate_video,
    screenshot_timestamps,
    take_screenshot,
)

SRC = Path("/videos/in.mp4")
OUT = Path("/videos/out.mp4")
H264 = VideoInfo(1920, 1080, 10.0, 0, "h264")
HEVC = VideoInfo(1920, 1080, 10.0, 0, "hevc")
VP9 = VideoInfo(1920, 1080, 10.0, 0, "vp9")
VP8 = VideoInfo(1920, 1080, 10.0, 0, "vp8")
MPEG4 = VideoInfo(1920, 1080, 10.0, 0, "mpeg4")


@pytest.mark.parametrize(
    "rotation,expected",
    [
        (0, (1920, 1080)),
        (180, (1920, 1080)),
        (-180, (1920, 1080)),
        (90, (1080, 1920)),
        (-90, (1080, 1920)),
    ],
)
def test_video_info_display_size(rotation, expected):
    assert VideoInfo(1920, 1080, 10.0, rotation).display_size == expected


def test_screenshot_timestamps_spreads_short_videos():
    # 10 seconds cannot fill a 30 second grid, so the minimum count is spread evenly
    stamps = screenshot_timestamps(10.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    assert stamps == [1.0, 3.0, 5.0, 7.0, 9.0]
    assert all(0 < t < 10.0 for t in stamps)


def test_screenshot_timestamps_uses_a_fixed_grid():
    # a trimmed copy keeps the timestamps of the original, which is what the grid is for
    full = screenshot_timestamps(300.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    trimmed = screenshot_timestamps(280.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    assert full == [15.0 + 30.0 * i for i in range(10)]
    assert trimmed == full[: len(trimmed)]


def test_screenshot_timestamps_respects_the_maximum():
    stamps = screenshot_timestamps(10000.0, interval=30.0, min_screenshots=5, max_screenshots=20)
    assert len(stamps) <= 20
    assert all(0 < t < 10000.0 for t in stamps)


@pytest.mark.parametrize("rotation_cw", sorted(ROTATE_FILTERS))
def test_build_rotate_command_reencodes_with_a_transpose_filter(rotation_cw):
    cmd = build_rotate_command(SRC, OUT, rotation_cw, H264, rotate_pixels=True)
    assert cmd[0] == "ffmpeg"
    assert cmd[-1] == OUT.as_posix()
    assert cmd[cmd.index("-filter:v:0") + 1] == ROTATE_FILTERS[rotation_cw]
    assert "-display_rotation:v:0" not in cmd
    # 180 degrees is two transposes, there is no single filter for it
    assert cmd.count("-filter:v:0") == 1


def test_a_pixel_rotation_always_writes_hevc_mp4():
    """One target means one encoder and one quality flag, whatever the source was."""
    for info in (H264, HEVC, VP9, MPEG4):
        cmd = build_rotate_command(SRC, OUT, 90, info, rotate_pixels=True)
        assert cmd[cmd.index("-c:v:0") + 1] == ROTATE_TARGET_ENCODER
        assert cmd[cmd.index("-crf") + 1] == str(DEFAULT_CRF)
        assert "hvc1" in cmd
        # the audio is converted because mp4 cannot carry vorbis or opus from a webm
        assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_a_pixel_rotation_refuses_another_container():
    for suffix in (".mkv", ".webm", ".avi"):
        with pytest.raises(ValueError, match="needs a .mp4 container"):
            build_rotate_command(SRC, Path(f"out{suffix}"), 90, H264, rotate_pixels=True)


@pytest.mark.parametrize(
    "name,rotate_pixels,expected",
    [
        ("clip.mp4", False, "clip.mp4"),
        ("clip.webm", False, "clip.webm"),
        ("clip.avi", False, "clip.avi"),
        ("clip.mp4", True, "clip.mp4"),
        ("clip.webm", True, "clip.mp4"),
        ("clip.avi", True, "clip.mp4"),
        ("clip.with.dots.mkv", True, "clip.with.dots.mp4"),
    ],
)
def test_rotate_target_file(name, rotate_pixels, expected):
    assert rotate_target_file(Path(name), rotate_pixels).name == expected


def test_build_rotate_command_defaults_to_the_container_rotation():
    cmd = build_rotate_command(SRC, OUT, 90, H264)
    assert "-display_rotation:v:0" in cmd
    assert "-vf" not in cmd
    assert cmd[cmd.index("-c") + 1] == "copy"


@pytest.mark.parametrize(
    "current_rotation,rotation_cw,expected",
    [
        # display rotation counts counter-clockwise, so a clockwise fix subtracts
        (0, 90, "270"),
        (0, 180, "180"),
        (0, 270, "90"),
        # a video that already declares a rotation keeps it and the fix is added on top
        (-90, 90, "180"),
        (-90, 270, "0"),
        (90, 90, "0"),
    ],
)
def test_build_rotate_command_container_rotation_math(current_rotation, rotation_cw, expected):
    info = VideoInfo(1920, 1080, 10.0, current_rotation, "h264")
    cmd = build_rotate_command(SRC, OUT, rotation_cw, info)
    assert cmd[cmd.index("-display_rotation:v:0") + 1] == expected
    # the rotation is an input option, so it has to come before -i
    assert cmd.index("-display_rotation:v:0") < cmd.index("-i")


def test_the_container_rotation_does_not_care_about_the_codec():
    for info in (H264, HEVC, VP9, VP8, MPEG4):
        cmd = build_rotate_command(SRC, OUT, 90, info)
        assert "-display_rotation:v:0" in cmd
        assert "-c:v" not in cmd


def test_build_rotate_command_refuses_a_container_without_rotation_support():
    for suffix in (".avi", ".mts"):
        with pytest.raises(ValueError, match="cannot store a display rotation"):
            build_rotate_command(SRC, Path(f"out{suffix}"), 90, H264)


def test_build_rotate_command_rejects_other_angles():
    for rotation_cw in (0, 45, 360, -90):
        with pytest.raises(ValueError, match="Cannot rotate"):
            build_rotate_command(SRC, OUT, rotation_cw, H264)


@pytest.mark.parametrize("pixels", [False, True])
def test_rotation_maps_all_streams(pixels):
    cmd = build_rotate_command(SRC, OUT, 90, H264, rotate_pixels=pixels)
    assert cmd[cmd.index("-map") + 1] == "0"
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-copy_unknown" in cmd


@pytest.mark.parametrize("pixels", [False, True])
def test_rotation_drops_streams_ffmpeg_cannot_copy(pixels):
    info = VideoInfo(1920, 1080, 10.0, 0, "h264", ((2, "mebx"), (5, "mebx")))
    cmd = build_rotate_command(SRC, OUT, 90, info, rotate_pixels=pixels)
    maps = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]
    assert maps == ["0", "-0:2", "-0:5"]


def test_a_failing_video_does_not_stop_the_batch(tmp_path, monkeypatch):
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for source in sources:
        source.write_bytes(b"original")
    monkeypatch.setattr(videotools, "probe_video", lambda _: H264)
    plan = plan_video_rotations(
        [(source, source.with_stem(f"{source.stem}_rot90"), 90) for source in sources]
    )

    def encode(cmd, **kwargs):
        if "a.mp4" in cmd[cmd.index("-i") + 1]:
            raise subprocess.CalledProcessError(234, cmd)
        Path(cmd[-1]).write_bytes(b"complete output")

    monkeypatch.setattr(videotools.subprocess, "run", encode)
    execute_video_rotations(plan, write=True)
    assert not plan[0].out_file.exists()
    assert plan[1].out_file.read_bytes() == b"complete output"
    assert [p.read_bytes() for p in sources] == [b"original", b"original"]
    assert not list(tmp_path.glob(".*.tmp.mp4"))


def test_plan_skips_an_unreadable_video(tmp_path, monkeypatch):
    sources = [tmp_path / "broken.mp4", tmp_path / "fine.mp4"]
    for source in sources:
        source.touch()

    def probe(video_file):
        if video_file.name == "broken.mp4":
            raise subprocess.CalledProcessError(1, "ffprobe", stderr="Invalid data")
        return H264

    monkeypatch.setattr(videotools, "probe_video", probe)
    plan = plan_video_rotations(
        [(source, tmp_path / "out" / source.name, 90) for source in sources]
    )
    assert [p.src_file.name for p in plan] == ["fine.mp4"]


def test_plan_chooses_method_and_extension_for_a_mixed_batch(tmp_path, monkeypatch):
    sources = [tmp_path / name for name in ("clip.mp4", "other.avi")]
    for source in sources:
        source.touch()
    monkeypatch.setattr(videotools, "probe_video", lambda _: H264)
    requests = [(source, tmp_path / "out" / source.name, 90) for source in sources]
    plan = plan_video_rotations(requests, rotate_pixels_if_needed=True)
    assert [(p.out_file.name, p.rotate_pixels) for p in plan] == [
        ("clip.mp4", False),
        ("other.mp4", True),
    ]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["duplicate", "existing", "unsupported", "missing"])
def test_plan_rejects_the_entire_batch_before_probing(tmp_path, monkeypatch, problem):
    source = tmp_path / "a.mp4"
    source.touch()
    second = tmp_path / "b.mp4"
    if problem != "missing":
        second.touch()
    first_target = tmp_path / "out" / "a.mp4"
    target = first_target if problem == "duplicate" else tmp_path / "b_rot90.mp4"
    if problem == "existing":
        target.write_bytes(b"keep existing output")
    if problem == "unsupported":
        target = tmp_path / "b_rot90.avi"

    def unexpected_probe(_):
        pytest.fail("The invalid batch must be rejected before probing or writing")

    monkeypatch.setattr(videotools, "probe_video", unexpected_probe)
    with pytest.raises((ValueError, FileExistsError, FileNotFoundError)):
        plan_video_rotations([(source, first_target, 90), (second, target, 90)])
    assert not first_target.parent.exists()
    if problem == "existing":
        assert target.read_bytes() == b"keep existing output"


def test_plan_detects_output_aliases(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "out", target_is_directory=True)
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for source in sources:
        source.touch()
    with pytest.raises(ValueError, match="both be rotated"):
        plan_video_rotations(
            [
                (sources[0], tmp_path / "out" / "video.mp4", 90),
                (sources[1], tmp_path / "alias" / "video.mp4", 90),
            ]
        )


def test_execution_rechecks_all_targets_before_callbacks(tmp_path, monkeypatch):
    sources = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for source in sources:
        source.touch()
    monkeypatch.setattr(videotools, "probe_video", lambda _: H264)
    plan = plan_video_rotations(
        [(source, source.with_stem(f"{source.stem}_rot90"), 90) for source in sources]
    )
    plan[1].out_file.write_bytes(b"appeared after planning")
    with pytest.raises(FileExistsError):
        execute_video_rotations(plan, write=True, before_rotate=lambda _: pytest.fail("No writes"))
    assert not plan[0].out_file.exists()


def test_rotation_failure_preserves_inputs_and_uses_unique_temporary_files(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    target = tmp_path / "target.mp4"
    other_input = tmp_path / "target.tmp.mp4"
    other_input.write_bytes(b"another input")
    temporary_files = []

    def fail_encoding(cmd, **kwargs):
        temporary = Path(cmd[-1])
        assert temporary.exists()  # already reserved exclusively
        temporary.write_bytes(b"partial encode")
        temporary_files.append(temporary)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(videotools.subprocess, "run", fail_encoding)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            rotate_video(source, target, 90, H264, delete_source=True)
    assert source.read_bytes() == b"original"
    assert other_input.read_bytes() == b"another input"
    assert not target.exists()
    assert len(set(temporary_files)) == 2
    assert all(
        p.name.startswith(".target.") and p.name.endswith(".tmp.mp4") for p in temporary_files
    )


@pytest.mark.parametrize("in_place", [False, True])
def test_successful_rotation_replaces_atomically_and_keeps_mtime(tmp_path, monkeypatch, in_place):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    original_ns = 1_700_000_000_123_456_789
    os.utime(source, ns=(original_ns, original_ns))
    target = source if in_place else tmp_path / "target.mp4"

    def encode(cmd, **kwargs):
        assert source.read_bytes() == b"original"
        Path(cmd[-1]).write_bytes(b"complete output")

    monkeypatch.setattr(videotools.subprocess, "run", encode)
    rotate_video(source, target, 90, H264, delete_source=True)
    assert target.read_bytes() == b"complete output"
    assert target.stat().st_mtime_ns == original_ns
    assert source.exists() == in_place
    assert not list(tmp_path.glob(".*.tmp.mp4"))


@pytest.fixture
def multistream_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are needed for stream-preservation coverage")
    source = tmp_path / "streams.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=128x96:rate=2:duration=1",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=192x128:rate=2:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=1",
            "-map",
            "0:v",
            "-map",
            "1:v",
            "-map",
            "2:a",
            "-map",
            "3:a",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=deu",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return source


def test_container_rotation_preserves_every_stream_and_packet(multistream_video):
    source = multistream_video
    target = source.with_name("rotated.mp4")
    rotate_video(source, target, 90, probe_video(source))

    def read_streams_and_packets(path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    before, after = [read_streams_and_packets(p) for p in (source, target)]
    assert len(before["streams"]) == len(after["streams"]) == 4
    for old, new in zip(before["streams"], after["streams"]):
        assert old["codec_name"] == new["codec_name"]
        assert old.get("tags", {}).get("language") == new.get("tags", {}).get("language")
        index = old["index"]
        assert [p["data_hash"] for p in before["packets"] if p["stream_index"] == index] == [
            p["data_hash"] for p in after["packets"] if p["stream_index"] == index
        ]
    assert probe_video(target).rotation == -90
    assert not any("rotation" in side for side in after["streams"][1].get("side_data_list", []))


def test_screenshot_uses_the_first_video_even_if_another_is_larger(multistream_video):
    image = take_screenshot(multistream_video, 0)
    assert image.size == (128, 96)
    red, _, blue = image.getpixel((0, 0))
    assert red > 200 and blue < 30


def test_pixel_rotation_keeps_all_audio_and_video_streams(multistream_video):
    target = multistream_video.with_name("pixels.mp4")
    rotate_video(multistream_video, target, 90, probe_video(multistream_video), rotate_pixels=True)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(target)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    streams = json.loads(result.stdout)["streams"]
    assert [s["codec_name"] for s in streams] == ["hevc", "h264", "aac", "aac"]
    assert [(s["width"], s["height"]) for s in streams if s["codec_type"] == "video"] == [
        (96, 128),
        (192, 128),
    ]
    assert all(not any("rotation" in d for d in s.get("side_data_list", [])) for s in streams)


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)


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
