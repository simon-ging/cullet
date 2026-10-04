import logging

import pytest

from cullet import cli_rotate_videos
from cullet import video as videotools


@pytest.fixture
def messages(caplog):
    caplog.set_level(logging.DEBUG)

    class Messages:
        def __iter__(self):
            return iter(caplog.messages)

    return Messages()


def _prepare_cli(tmp_path, monkeypatch, filenames):
    paths = [tmp_path / filename for filename in filenames]
    for path in paths:
        path.touch()
    module = cli_rotate_videos
    monkeypatch.setattr(module, "configure_logging", lambda args: None)
    monkeypatch.setattr(videotools, "probe_video", lambda _: videotools.VideoInfo(64, 48, 1, 0))
    monkeypatch.setattr(
        videotools, "rotate_video", lambda *a, **kw: pytest.fail("No video should be written")
    )
    argv = ["cullet-rotate-videos", *map(str, paths), "-R", "90"]
    return module, argv


def test_dry_run_lists_each_method_and_the_summary(tmp_path, monkeypatch, messages):
    module, argv = _prepare_cli(tmp_path, monkeypatch, ["clip.mp4", "other.avi"])
    monkeypatch.setattr("sys.argv", [*argv, "-A"])
    module.main()
    assert any("(container)" in line and "clip.mp4" in line for line in messages)
    assert any("(re-encode to hevc mp4)" in line and "other.avi" in line for line in messages)
    assert any(
        "'rotate,container': 1" in line and "'rotate,reencode': 1" in line for line in messages
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4", "other.avi"]


def test_collision_is_rejected_before_the_first_write(tmp_path, monkeypatch):
    module, argv = _prepare_cli(tmp_path, monkeypatch, ["a.webm", "a.mov"])
    monkeypatch.setattr("sys.argv", [*argv, "-P", "-w"])
    with pytest.raises(ValueError, match="both be rotated"):
        module.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mov", "a.webm"]


def test_existing_target_is_rejected_before_the_first_write(tmp_path, monkeypatch):
    module, argv = _prepare_cli(tmp_path, monkeypatch, ["a.webm", "b.mov"])
    (tmp_path / "b.mp4").write_bytes(b"existing video")
    monkeypatch.setattr("sys.argv", [*argv, "-P", "-O", "-w"])
    with pytest.raises(FileExistsError):
        module.main()
    assert not (tmp_path / "a.mp4").exists()
    assert (tmp_path / "b.mp4").read_bytes() == b"existing video"


def test_conflicting_flags_are_rejected(tmp_path, monkeypatch):
    module, argv = _prepare_cli(tmp_path, monkeypatch, ["clip.mp4"])
    monkeypatch.setattr("sys.argv", [*argv, "-P", "-A"])
    with pytest.raises(ValueError, match="not both"):
        module.main()


def test_the_cli_executes_the_shared_plan(tmp_path, monkeypatch):
    module, argv = _prepare_cli(tmp_path, monkeypatch, ["clip.mp4", "other.avi"])
    calls = []
    monkeypatch.setattr(
        videotools, "rotate_video", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr("sys.argv", [*argv, "-A", "-O", "-w", "-Q", "23", "--preset", "fast"])
    module.main()
    assert [(a[0].name, a[1].name, a[2], kw) for a, kw in calls] == [
        (
            "clip.mp4",
            "clip.mp4",
            90,
            {
                "crf": 23,
                "preset": "fast",
                "rotate_pixels": False,
                "delete_source": True,
            },
        ),
        (
            "other.avi",
            "other.mp4",
            90,
            {
                "crf": 23,
                "preset": "fast",
                "rotate_pixels": True,
                "delete_source": True,
            },
        ),
    ]
