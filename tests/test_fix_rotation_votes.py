import pytest

# torch is an optional dependency, the full extra
torch = pytest.importorskip("torch")

from cullet.cli_fix_rotation_videos import vote_on_rotation  # noqa: E402
from cullet.rotation import N_ROTATIONS  # noqa: E402

MIN_CONFIDENCE = 0.9


def _probs(votes: list[tuple[int, float]]) -> torch.Tensor:
    """(n_screenshots, 4) probabilities from (rotation, confidence) pairs."""
    out = torch.zeros(len(votes), N_ROTATIONS)
    for row, (rotation, confidence) in enumerate(votes):
        out[row] = (1 - confidence) / (N_ROTATIONS - 1)
        out[row, rotation] = confidence
    return out


def test_all_screenshots_agree():
    rotation, agreement, n_votes = vote_on_rotation(_probs([(2, 0.99)] * 5), MIN_CONFIDENCE)
    assert (rotation, agreement, n_votes) == (2, 1.0, 5)


def test_a_single_outlier_does_not_win():
    votes = [(3, 0.99), (3, 0.99), (3, 0.99), (1, 0.99)]
    rotation, agreement, n_votes = vote_on_rotation(_probs(votes), MIN_CONFIDENCE)
    assert rotation == 3
    assert agreement == 0.75
    assert n_votes == 4


def test_agreement_counts_all_screenshots_not_only_the_voters():
    # two confident screenshots agree, the other two are too unsure to vote. agreeing on half
    # of the video is not enough, so the default threshold of 0.75 rejects this
    votes = [(1, 0.99), (1, 0.99), (1, 0.5), (1, 0.5)]
    rotation, agreement, n_votes = vote_on_rotation(_probs(votes), MIN_CONFIDENCE)
    assert rotation == 1
    assert agreement == 0.5
    assert n_votes == 2


def test_no_confident_screenshot_means_no_rotation():
    rotation, agreement, n_votes = vote_on_rotation(_probs([(2, 0.4)] * 4), MIN_CONFIDENCE)
    assert (rotation, agreement, n_votes) == (0, 0.0, 0)


def test_a_split_video_does_not_reach_the_default_threshold():
    votes = [(0, 0.99), (0, 0.99), (3, 0.99), (3, 0.99)]
    _, agreement, _ = vote_on_rotation(_probs(votes), MIN_CONFIDENCE)
    assert agreement == 0.5


@pytest.mark.parametrize("first,second", [(0, 2), (2, 0), (1, 3)])
def test_a_tie_uses_the_first_encountered_winner(first, second):
    votes = [(first, 0.99), (second, 0.99)]
    rotation, agreement, _ = vote_on_rotation(_probs(votes), MIN_CONFIDENCE)
    assert rotation == first
    assert agreement == 0.5
