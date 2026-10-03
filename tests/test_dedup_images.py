import pytest

# torch is an optional dependency, the dedup extra
torch = pytest.importorskip("torch")

from cullet.dedup.image_groups import DedupImageConfig  # noqa: E402
from cullet.dedup.images import make_groups  # noqa: E402


def _entry(emb, width, height, size):
    return {
        "emb": torch.tensor(emb, dtype=torch.float32),
        "width": width,
        "height": height,
        "size": size,
    }


def test_make_groups_orders_keep_first(tmp_path):
    rel_files = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    entries = {
        # a and b are near duplicates, b has clearly more pixels so it is the keep
        "a.jpg": _entry([1.0, 0.0], 1000, 1000, 100),
        "b.jpg": _entry([0.99, 0.1], 2000, 2000, 300),
        # d is bigger in bytes only within the tolerance, so c wins by name
        "c.jpg": _entry([0.0, 1.0], 500, 500, 100),
        "d.jpg": _entry([0.1, 0.99], 500, 500, 105),
    }
    pairs = [(0, 1, 0.99), (2, 3, 0.99)]
    groups = make_groups(tmp_path, rel_files, entries, pairs, DedupImageConfig())
    assert [[m.rel_file for m in g.members] for g in groups] == [
        ["b.jpg", "a.jpg"],
        ["c.jpg", "d.jpg"],
    ]
    assert groups[0].keep.path == (tmp_path / "b.jpg").absolute()
    assert [m.rel_file for m in groups[0].removals] == ["a.jpg"]
    assert abs(groups[0].members[1].sim - 0.99) < 0.02
    assert groups[1].members[0].size == 100
