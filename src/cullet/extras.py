"""
Optional dependencies. The viewer and the downscaling run without them, finding duplicates and
fixing rotations needs them.
"""

import importlib.util

# import name of every package of the full extra in pyproject.toml
FULL_PACKAGES = ["torch", "torchvision", "tqdm", "numpy"]


def require_full_extra() -> None:
    """Exit with a message that says what to install, instead of an ImportError from somewhere
    inside the program. Call this before importing the modules that need the extra."""
    missing = [name for name in FULL_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            f"This needs the packages {', '.join(missing)}, which are not installed. "
            f"They come with the full extra:\n\n"
            f'    uv tool install "cullet[full]"\n\n'
            f'or, inside an environment: pip install "cullet[full]"'
        )


def fix_rotation_images() -> None:
    """Entry point. The module behind it imports torch, so the check comes first."""
    require_full_extra()
    from cullet.cli_fix_rotation_images import main

    main()


def fix_rotation_videos() -> None:
    """Entry point. The module behind it imports torch, so the check comes first."""
    require_full_extra()
    from cullet.cli_fix_rotation_videos import main

    main()
