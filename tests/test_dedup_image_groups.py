from pathlib import Path

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
