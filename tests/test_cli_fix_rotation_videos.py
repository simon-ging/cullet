import logging
from types import SimpleNamespace

import pytest

# torch is an optional dependency, the full extra
torch = pytest.importorskip("torch")

from cullet import cli_fix_rotation_videos as fix_rotation_videos  # noqa: E402
from cullet import video as videotools  # noqa: E402


@pytest.fixture
def messages(caplog):
    caplog.set_level(logging.DEBUG)

    class Messages:
        def __iter__(self):
            return iter(caplog.messages)

    return Messages()


def _prepare_cli(name, tmp_path, monkeypatch, filenames):
    paths = [tmp_path / filename for filename in filenames]
    for path in paths:
        path.touch()
    module = fix_rotation_videos
    monkeypatch.setattr(module, "configure_logging", lambda args: None)
    monkeypatch.setattr(videotools, "probe_video", lambda _: videotools.VideoInfo(64, 48, 1, 0))
    monkeypatch.setattr(
        videotools, "rotate_video", lambda *a, **kw: pytest.fail("No video should be written")
    )
    monkeypatch.setattr(
        fix_rotation_videos, "write_check_image", lambda *a, **kw: pytest.fail("No check image")
    )
    entries = {
        filename: {
            "logp": torch.tensor([[0.01, 0.97, 0.01, 0.01]] * 5),
            "width": 64,
            "height": 48,
            "duration": 1,
        }
        for filename in filenames
    }
    monkeypatch.setattr(
        fix_rotation_videos, "compute_predictions", lambda *a: (filenames, entries, {})
    )
    monkeypatch.setattr(fix_rotation_videos, "combine_rotation_views", lambda p: p)
    argv = ["fix_rotation_videos", str(tmp_path)]
    return module, argv


@pytest.mark.parametrize("name", ["detected"])
def test_dry_run_lists_each_method_and_the_summary(tmp_path, monkeypatch, messages, name):
    module, argv = _prepare_cli(name, tmp_path, monkeypatch, ["clip.mp4", "other.avi"])
    monkeypatch.setattr("sys.argv", [*argv, "-A"])
    module.main()
    assert any("(container)" in line and "clip.mp4" in line for line in messages)
    assert any("(re-encode to hevc mp4)" in line and "other.avi" in line for line in messages)
    assert any(
        "'rotate,container': 1" in line and "'rotate,reencode': 1" in line for line in messages
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4", "other.avi"]


@pytest.mark.parametrize("name", ["detected"])
def test_collision_is_rejected_before_the_first_write(tmp_path, monkeypatch, name):
    module, argv = _prepare_cli(name, tmp_path, monkeypatch, ["a.webm", "a.mov"])
    monkeypatch.setattr("sys.argv", [*argv, "-P", "-w"])
    with pytest.raises(ValueError, match="both be rotated"):
        module.main()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mov", "a.webm"]


@pytest.mark.parametrize("name", ["detected"])
def test_existing_target_is_rejected_before_the_first_write(tmp_path, monkeypatch, name):
    module, argv = _prepare_cli(name, tmp_path, monkeypatch, ["a.webm", "b.mov"])
    (tmp_path / "b.mp4").write_bytes(b"existing video")
    monkeypatch.setattr("sys.argv", [*argv, "-P", "-O", "-w"])
    with pytest.raises(FileExistsError):
        module.main()
    assert not (tmp_path / "a.mp4").exists()
    assert (tmp_path / "b.mp4").read_bytes() == b"existing video"


@pytest.mark.parametrize("name", ["detected"])
def test_conflicting_flags_are_rejected(tmp_path, monkeypatch, name):
    module, argv = _prepare_cli(name, tmp_path, monkeypatch, ["clip.mp4"])
    monkeypatch.setattr("sys.argv", [*argv, "-P", "-A"])
    with pytest.raises(ValueError, match="not both"):
        module.main()


@pytest.mark.parametrize("name", ["detected"])
def test_both_clis_execute_the_shared_plan(tmp_path, monkeypatch, name):
    module, argv = _prepare_cli(name, tmp_path, monkeypatch, ["clip.mp4", "other.avi"])
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


@pytest.mark.parametrize("agreement,accepted", [(0.5, True), (0.51, False)])
def test_tied_vote_obeys_the_inclusive_threshold(
    tmp_path, monkeypatch, messages, agreement, accepted
):
    module, argv = _prepare_cli("detected", tmp_path, monkeypatch, ["clip.mp4"])
    monkeypatch.setattr(
        module,
        "combine_rotation_views",
        lambda _: torch.tensor([[0.01, 0.01, 0.97, 0.01], [0.97, 0.01, 0.01, 0.01]]),
    )
    monkeypatch.setattr("sys.argv", [*argv, "-a", str(agreement)])
    module.main()
    assert any("Rotate 180° cw (container)" in line for line in messages) == accepted


def test_detection_skips_generated_temporary_files(tmp_path, monkeypatch):
    module = fix_rotation_videos
    args = module.Args(input_dir=tmp_path)
    props = SimpleNamespace(st_size=1, st_mtime=0)
    monkeypatch.setattr(
        module,
        "index_files",
        lambda *a, **kw: {"clip.mp4": props, "clip.tmp.mp4": props, ".clip.random.tmp.mp4": props},
    )
    entry = {"screenshot_params": module.screenshot_params(args)}
    cache = SimpleNamespace(get=lambda *a: entry)
    monkeypatch.setattr(module, "EmbeddingCache", lambda *a, **kw: cache)
    files, _, _ = module.compute_predictions(tmp_path, args, None)
    assert files == ["clip.mp4", "clip.tmp.mp4"]


def test_old_stream_selection_predictions_are_recomputed(tmp_path, monkeypatch):
    module = fix_rotation_videos
    args = module.Args(input_dir=tmp_path)
    monkeypatch.setattr(
        module, "index_files", lambda *a, **kw: {"clip.mp4": SimpleNamespace(st_size=1, st_mtime=0)}
    )
    old_entry = {"screenshot_params": ("v1", 30.0, 5, 20)}
    monkeypatch.setattr(
        module, "EmbeddingCache", lambda *a, **kw: SimpleNamespace(get=lambda *a: old_entry)
    )

    def classifier_needed(*args):
        raise RuntimeError("Recompute predictions for the first video stream")

    monkeypatch.setattr(module, "OrientationClassifier", classifier_needed)
    with pytest.raises(RuntimeError, match="Recompute predictions"):
        module.compute_predictions(tmp_path, args, None)
