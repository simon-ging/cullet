"""
Optional dependencies. The viewer runs without them, finding duplicates needs them.
"""

import importlib.util

# import name of every package of the dedup extra in pyproject.toml
DEDUP_PACKAGES = ["torch", "torchvision", "tqdm", "numpy", "pathspec"]


def require_dedup_extra() -> None:
    """Exit with a message that says what to install, instead of an ImportError from somewhere
    inside the program. Call this before importing the modules that need the extra."""
    missing = [name for name in DEDUP_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            f"Finding duplicates needs the packages {', '.join(missing)}, which are not "
            f"installed. They come with the dedup extra:\n\n"
            f'    uv tool install "cullet[dedup]"\n\n'
            f'or, inside an environment: pip install "cullet[dedup]"'
        )
