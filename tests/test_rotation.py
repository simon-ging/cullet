import numpy as np
import pytest
from PIL import Image

# torch is an optional dependency, the full extra
torch = pytest.importorskip("torch")

from cullet.rotation import (  # noqa: E402
    N_ROTATIONS,
    TRANSPOSES,
    combine_rotation_views,
    evaluate_rotations,
    rotate_image,
)


def _make_log_probs(true_rotation: int, confidence: float = 0.9) -> torch.Tensor:
    """(1, 4, 4) log probs of a classifier that predicts view k as true_rotation + k."""
    probs = torch.full((1, N_ROTATIONS, N_ROTATIONS), (1 - confidence) / (N_ROTATIONS - 1))
    for k in range(N_ROTATIONS):
        probs[0, k, (true_rotation + k) % N_ROTATIONS] = confidence
    return probs.log()


@pytest.mark.parametrize("true_rotation", range(N_ROTATIONS))
def test_combine_rotation_views(true_rotation):
    probs = combine_rotation_views(_make_log_probs(true_rotation))
    assert probs.shape == (1, N_ROTATIONS)
    assert int(probs.argmax()) == true_rotation
    # 4 agreeing views are more confident than a single one
    assert probs[0, true_rotation] > 0.9
    assert torch.allclose(probs.sum(), torch.tensor(1.0))


def test_combine_rotation_views_disagreement():
    # view 0 says upright, the other views say rotated by 2: the majority wins
    log_probs = _make_log_probs(2)
    log_probs[0, 0] = _make_log_probs(0)[0, 0]
    probs = combine_rotation_views(log_probs)
    assert int(probs.argmax()) == 2


def test_evaluate_rotations():
    log_probs = torch.cat([_make_log_probs(0, 0.9), _make_log_probs(0, 0.4)])
    metrics = evaluate_rotations(log_probs, min_confidence=0.5)
    assert metrics["single_acc"] == 1.0
    assert metrics["combined_acc"] == 1.0
    assert 0 < metrics["confident_frac"] <= 1.0
    assert metrics["confident_acc"] == 1.0
    # a classifier that always predicts the same class is right for 1 of 4 rotations
    constant = torch.full((3, N_ROTATIONS, N_ROTATIONS), -10.0)
    constant[:, :, 0] = 0.0
    metrics = evaluate_rotations(constant, min_confidence=0.5)
    assert metrics["single_acc"] == 0.25
    assert metrics["combined_acc"] == 0.25


def test_rotate_image_matches_torch_rot90():
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(array)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    for k in range(N_ROTATIONS):
        rotated = rotate_image(image, k)
        expected = torch.rot90(tensor, k, dims=(1, 2)).permute(1, 2, 0).numpy()
        assert np.array_equal(np.asarray(rotated), expected), f"k={k}"
    # rotating back by the complement gives the original
    for k in range(N_ROTATIONS):
        back = rotate_image(rotate_image(image, k), N_ROTATIONS - k)
        assert np.array_equal(np.asarray(back), array)
    assert len(TRANSPOSES) == N_ROTATIONS
