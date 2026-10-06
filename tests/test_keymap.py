import pytest

from cullet.actions import Actions
from cullet.keymap import (
    ACTION_GROUPS,
    DEDUP_KEYMAP,
    DEFAULT_KEYMAP,
    make_tag_keymap,
    make_target_keymap,
    menu_entries,
    split_action,
)


def test_every_binding_has_an_action():
    for keymap in (DEFAULT_KEYMAP, DEDUP_KEYMAP, make_target_keymap(9), make_tag_keymap(9)):
        for value in keymap.values():
            action_name, _args = split_action(value)
            assert getattr(Actions, action_name).__doc__, f"{action_name} needs a docstring"


def test_every_action_is_in_one_menu_group():
    grouped = [name for names in ACTION_GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped)), "an action is in two groups"
    actions = {
        name
        for name, member in vars(Actions).items()
        if callable(member) and not name.startswith("_")
    }
    assert set(grouped) == actions


def test_menu_entries():
    menus = dict(menu_entries(DEFAULT_KEYMAP, Actions(None)))
    assert ("next_image", "Next image", "Space PgDown D") in menus["&Navigate"]
    assert "&Review" not in menus, "nothing of the review is bound outside of it"
    bindings = [binding for entries in menus.values() for binding, _label, _keys in entries]
    assert sorted(bindings) == sorted(set(DEFAULT_KEYMAP.values()))

    menus = dict(menu_entries(DEDUP_KEYMAP, Actions(None)))
    assert ("delete_member:3", "Move image 3 of the group to the trash", "3") in menus["&Review"]
    assert "&Listing" not in menus

    with pytest.raises(AssertionError):
        menu_entries({"X": "no_such_action"}, Actions(None))


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
