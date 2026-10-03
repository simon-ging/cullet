from pathlib import Path

import torch
from attrs import define

from typedparser import TypedParser, VerboseQuietArgs

from cullet.dedup.image_groups import (
    DEDUP_IMAGE_ARG_SPECS,
    DedupImageArgs,
    DedupImageConfig,
    DuplicateMember,
    dedup_config_from_args,
    describe,
    make_dedup_image_args_class,
)
from cullet.dedup.images import make_groups

PrefixedArgs = make_dedup_image_args_class("PrefixedArgs", prefix="dedup_")


@define
class PlainArgs(DedupImageArgs, VerboseQuietArgs):
    pass


@define
class ViewerArgs(PrefixedArgs, VerboseQuietArgs):
    pass


def test_prefixed_args_give_same_config():
    plain = TypedParser.create_parser(PlainArgs).parse_args(["-t", "0.8", "-r", "-m", "m.pt"])
    prefixed = TypedParser.create_parser(ViewerArgs).parse_args(
        ["--dedup_threshold", "0.8", "--dedup_non_recursive", "--dedup_model_file", "m.pt"]
    )
    config = dedup_config_from_args(plain)
    assert config == dedup_config_from_args(prefixed, prefix="dedup_")
    assert config == DedupImageConfig(recursive=False, threshold=0.8, model_file=Path("m.pt"))
    # every setting exists under both names
    for name in DEDUP_IMAGE_ARG_SPECS:
        assert hasattr(plain, name) and hasattr(prefixed, f"dedup_{name}")
    assert dedup_config_from_args(TypedParser.create_parser(PlainArgs).parse_args([])) == (
        DedupImageConfig()
    )


def test_describe():
    member = DuplicateMember(Path("/a.jpg"), "a.jpg", 4000, 1440, int(1.38 * 1024**2), 0.999)
    assert describe(member) == " 4000x1440    1.38MB"


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
