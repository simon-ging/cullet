import pytest

from cullet.cli_downscale_images import _over_trigger


@pytest.mark.parametrize("side", [1, 500, 1080, 4000])
def test_without_a_trigger_every_image_is_resized(side):
    assert _over_trigger(side, None)


def test_a_trigger_only_lets_longer_sides_through():
    assert not _over_trigger(1499, 1500)
    # the threshold itself is not over it, "exceeds" is strict
    assert not _over_trigger(1500, 1500)
    assert _over_trigger(1501, 1500)


def test_mpo_counts_as_jpeg_for_the_rewrite_decision():
    """A multi frame jpeg must not be rewritten just to become a plain jpeg."""
    from cullet.cli_downscale_images import EQUIVALENT_FORMATS

    assert EQUIVALENT_FORMATS["MPO"] == "JPEG"
    # anything else keeps its own identity, so a real conversion still triggers a rewrite
    for source in ("TIFF", "BMP", "WEBP", "PNG"):
        assert EQUIVALENT_FORMATS.get(source, source) == source
