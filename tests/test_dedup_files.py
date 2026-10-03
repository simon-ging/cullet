import pytest

pytest.importorskip("pathspec")  # part of the dedup extra

from cullet.dedup.files import (  # noqa: E402
    PathSpecArgs,
    filter_rel_files,
    index_files,
    remove_duplicates,
)

FILES = ["a.jpg", "b.png", "sub/c.jpg", "sub/deep/d.jpg", "other/e.jpg"]


@pytest.fixture
def folder(tmp_path):
    folder = tmp_path / "photos"
    for rel_file in FILES:
        (folder / rel_file).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel_file).write_bytes(b"x")
    return folder


def remaining(folder) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def test_filter_rel_files():
    assert filter_rel_files(FILES) == FILES
    assert filter_rel_files(FILES, include_git=["*.png"]) == ["b.png"]
    assert filter_rel_files(FILES, exclude_git=["sub/"]) == ["a.jpg", "b.png", "other/e.jpg"]
    # a leading slash anchors the pattern at the folder
    assert filter_rel_files(FILES, include_git=["/*.jpg"]) == ["a.jpg"]
    assert filter_rel_files(FILES, exclude_regex=[r"deep|other"]) == ["a.jpg", "b.png", "sub/c.jpg"]
    assert filter_rel_files(FILES, include_regex=[r"\.jpg$"], exclude_git=["sub/"]) == [
        "a.jpg",
        "other/e.jpg",
    ]


def test_filter_with_gitignore_file(tmp_path):
    ignore_file = tmp_path / "ignore"
    ignore_file.write_text("*.png\nother/\n")
    kept = filter_rel_files(FILES, exclude_gitignore_file=ignore_file)
    assert kept == ["a.jpg", "sub/c.jpg", "sub/deep/d.jpg"]


def test_index_files(folder):
    assert sorted(index_files(folder, recursive=True)) == sorted(FILES)
    assert sorted(index_files(folder, recursive=False)) == ["a.jpg", "b.png"]
    args = PathSpecArgs(exclude_git=["sub/"], include_git=["*.jpg"])
    index = index_files(folder, recursive=True, pathspec_args=args)
    assert sorted(index) == ["a.jpg", "other/e.jpg"]
    assert index["a.jpg"].st_size == 1


def test_remove_duplicates_dry_run_changes_nothing(folder):
    remove_duplicates(folder, ["a.jpg", "sub/c.jpg"], None, write=False)
    assert remaining(folder) == sorted(FILES)


def test_remove_duplicates_into_quarantine(folder, tmp_path):
    quarantine = tmp_path / "quarantine"
    remove_duplicates(folder, ["a.jpg", "sub/c.jpg"], quarantine, write=True)
    assert remaining(folder) == ["b.png", "other/e.jpg", "sub/deep/d.jpg"]
    assert remaining(quarantine) == ["a.jpg", "sub/c.jpg"]


def test_remove_duplicates_into_system_trash(folder, tmp_path, monkeypatch):
    # the trash of the desktop lives in the data dir of the user, point it into the test dir
    data_home = tmp_path / "data_home"
    data_home.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", data_home.as_posix())
    remove_duplicates(folder, ["a.jpg", "sub/c.jpg", "b.png"], None, True, delete_only=["*.jpg"])
    assert remaining(folder) == ["b.png", "other/e.jpg", "sub/deep/d.jpg"]
    assert remaining(data_home / "Trash" / "files") == ["a.jpg", "c.jpg"]


def test_remove_duplicates_unlink(folder):
    remove_duplicates(folder, ["a.jpg"], None, write=True, unlink=True)
    assert "a.jpg" not in remaining(folder)
