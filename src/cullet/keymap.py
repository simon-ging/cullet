"""
Default key bindings, geeqie-like. Keys are Qt key sequence strings, values are method names
of cullet.actions.Actions, optionally with one integer argument after a colon
("delete_member:3"). The help overlay is generated from this table and the docstrings of those
methods, so every new binding needs a documented action.
"""

DEFAULT_KEYMAP: dict[str, str] = {
    "Space": "next_image",
    "PgDown": "next_image",
    "Backspace": "prev_image",
    "PgUp": "prev_image",
    "Home": "first_image",
    "End": "last_image",
    "Left": "arrow_left",
    "Right": "arrow_right",
    "Up": "arrow_up",
    "Down": "arrow_down",
    "[": "rotate_left",
    "]": "rotate_right",
    "Del": "delete_current",
    "F2": "rename_current",
    "U": "undo",
    "I": "toggle_zoom_filter",
    "Z": "zoom_actual",
    "X": "zoom_fit",
    "+": "zoom_in",
    "=": "zoom_in",
    "-": "zoom_out",
    "W": "nav_up",
    "A": "prev_image",
    "S": "nav_down",
    "D": "next_image",
    "M": "toggle_slideshow",
    ",": "slideshow_slower",
    ".": "slideshow_faster",
    "Shift+M": "toggle_slideshow_random",
    "R": "toggle_recursive",
    "N": "cycle_sort_order",
    "F": "toggle_fullscreen",
    "G": "toggle_grid",
    "T": "toggle_thumbnails",
    "L": "toggle_left_panel",
    "B": "toggle_bottom",
    "O": "toggle_info",
    "H": "toggle_help",
    "Q": "quit",
    "Escape": "exit_fullscreen",
}

# duplicate review: J/K step through the groups, 1-9 trash one member, P accepts the proposal.
# Recursive and sort order do not apply to a group listing.
DEDUP_KEYMAP: dict[str, str] = {
    **{k: v for k, v in DEFAULT_KEYMAP.items() if k not in ("R", "N")},
    "J": "next_group",
    "K": "prev_group",
    "P": "accept_proposal",
    **{str(n): f"delete_member:{n}" for n in range(1, 10)},
}


def make_target_keymap(n_targets: int) -> dict[str, str]:
    """Keys 1-9 move the current image into the target folder of that number."""
    assert 0 < n_targets <= 9, f"Between 1 and 9 target folders, got {n_targets}"
    return {str(n): f"move_to_target:{n}" for n in range(1, n_targets + 1)}


def make_tag_keymap(n_tags: int, offset: int = 0) -> dict[str, str]:
    """Number keys toggle a tag in the file name. The offset is how many number keys the target
    folders took already, so folders and tags can be used together."""
    assert (
        0 < n_tags and offset + n_tags <= 9
    ), f"There are 9 number keys, {offset} are taken and {n_tags} tags do not fit"
    return {str(offset + n): f"toggle_tag:{n}" for n in range(1, n_tags + 1)}


# these only act while the image or the grid has the focus, so they keep their usual meaning
# in the folder tree and the file list
VIEW_KEYS = {"Left", "Right", "Up", "Down"}


def split_action(value: str) -> tuple[str, list[int]]:
    """'delete_member:3' -> ('delete_member', [3]), 'next_image' -> ('next_image', [])."""
    name, _, arg = value.partition(":")
    return name, [int(arg)] if arg else []


MOUSE_BINDINGS: list[tuple[str, str]] = [
    ("Wheel", "next/prev image"),
    ("Ctrl+Wheel", "zoom, on thumbnails: cell size"),
    ("Shift+Wheel", "on thumbnails: scroll"),
    ("Drag", "pan"),
    ("Click", "on thumbnails: select"),
    ("Double click", "on image: fullscreen, on thumbnails: open"),
]


def format_help(keymap: dict[str, str], actions: object) -> str:
    """One line per action: its keys, then the docstring of the action method."""
    keys_per_action: dict[str, list[str]] = {}
    for key, value in keymap.items():
        action_name, _args = split_action(value)
        keys_per_action.setdefault(action_name, []).append(key)
    rows = []
    for action_name, keys in keys_per_action.items():
        doc = getattr(type(actions), action_name).__doc__
        assert doc, f"Action {action_name} needs a docstring, it is shown in the help overlay"
        rows.append((" ".join(keys), " ".join(doc.split())))
    rows += MOUSE_BINDINGS
    width = max(len(keys) for keys, _ in rows)
    return "\n".join(f"{keys:<{width}}  {description}" for keys, description in rows)
