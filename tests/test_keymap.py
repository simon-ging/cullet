import pytest

from cullet.actions import Actions
from cullet.keymap import (
    DEDUP_KEYMAP,
    DEFAULT_KEYMAP,
    make_tag_keymap,
    make_target_keymap,
    split_action,
)


def test_every_binding_has_an_action():
    for keymap in (DEFAULT_KEYMAP, DEDUP_KEYMAP, make_target_keymap(9), make_tag_keymap(9)):
        for value in keymap.values():
            action_name, _args = split_action(value)
            assert getattr(Actions, action_name).__doc__, f"{action_name} needs a docstring"


def test_wasd_navigates():
    assert DEFAULT_KEYMAP["A"] == "prev_image"
    assert DEFAULT_KEYMAP["D"] == "next_image"
    assert DEFAULT_KEYMAP["W"] == "nav_up"
    assert DEFAULT_KEYMAP["S"] == "nav_down"
    assert DEFAULT_KEYMAP["M"] == "toggle_slideshow", "w/a/s/d pushed the slideshow off S"


def test_target_keymap():
    assert make_target_keymap(2) == {"1": "move_to_target:1", "2": "move_to_target:2"}
    assert split_action("move_to_target:2") == ("move_to_target", [2])
    with pytest.raises(AssertionError):
        make_target_keymap(10)


def test_tag_keymap():
    assert make_tag_keymap(2) == {"1": "toggle_tag:1", "2": "toggle_tag:2"}
    assert make_tag_keymap(2, offset=3) == {"4": "toggle_tag:1", "5": "toggle_tag:2"}
    assert not set(make_target_keymap(3)) & set(make_tag_keymap(6, offset=3)), "9 keys, no overlap"
    with pytest.raises(AssertionError):
        make_tag_keymap(7, offset=3)
