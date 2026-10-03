from pathlib import Path

from cullet.dedup.image_groups import DedupImageResult, DuplicateGroup, DuplicateMember
from cullet.dedup_review import DedupReview
from cullet.keymap import DEDUP_KEYMAP, DEFAULT_KEYMAP, split_action


def make_review(folder: Path) -> DedupReview:
    def member(name, sim=0.95):
        path = folder / name
        path.write_bytes(b"x")
        return DuplicateMember(path, name, 100, 50, 1024, sim)

    groups = [
        DuplicateGroup([member("keep1.jpg"), member("del1a.jpg", 0.99), member("del1b.jpg")]),
        DuplicateGroup([member("keep2.jpg"), member("del2.jpg")]),
    ]
    return DedupReview(DedupImageResult(groups, n_images=10, n_pairs=3, broken_files={}))


def test_review_labels_and_resolution(tmp_path):
    review = make_review(tmp_path)
    assert review.summary() == "DEDUP 10 images, 3 similar pairs forming 2 duplicate groups"
    assert review.files(0) == [tmp_path / n for n in ["keep1.jpg", "del1a.jpg", "del1b.jpg"]]
    labels = review.labels(0)
    assert labels[0].startswith("KEEP ") and labels[0].endswith("keep1.jpg")
    assert labels[1].startswith("DEL ") and "sim=0.990" in labels[1]
    assert review.member_note(tmp_path / "del1a.jpg") == "  100x50      0.00MB sim=0.990"
    assert review.n_unresolved() == 2
    assert review.group_summary(0) == "     3/3 imgs  sim 0.990  keep1.jpg"

    (tmp_path / "del1a.jpg").unlink()
    assert review.files(0) == [tmp_path / "keep1.jpg", tmp_path / "del1b.jpg"]
    assert not review.is_resolved(0)
    (tmp_path / "del1b.jpg").unlink()
    assert review.is_resolved(0)
    assert review.n_unresolved() == 1
    assert review.group_summary(0).startswith("done 1/3 imgs")
    assert review.next_unresolved(0, 1) == 1
    assert review.next_unresolved(1, 1) is None
    assert review.next_unresolved(1, -1) is None
    assert review.group_of(tmp_path / "del2.jpg") == 1
    assert review.group_of(tmp_path / "other.jpg") is None


def test_dedup_keymap():
    assert split_action("delete_member:3") == ("delete_member", [3])
    assert split_action("next_image") == ("next_image", [])
    assert DEDUP_KEYMAP["P"] == "accept_proposal"
    assert DEDUP_KEYMAP["G"] == "toggle_grid"
    assert DEDUP_KEYMAP["5"] == "delete_member:5"
    assert "R" not in DEDUP_KEYMAP and "N" not in DEDUP_KEYMAP
    assert DEFAULT_KEYMAP["G"] == "toggle_grid"
