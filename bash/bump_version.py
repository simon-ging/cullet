"""
Make sure the version in src/cullet/__init__.py is higher than the one on PyPI, which refuses
an upload of a version that exists already. If it is not, it is set to the PyPI version with
the patch number increased by one. A version that is already higher is left alone, so a minor
or major bump done by hand survives.
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = "cullet"
INIT_FILE = Path(__file__).parent.parent / "src" / PACKAGE / "__init__.py"
RE_VERSION = re.compile(r'^__version__\s*=\s*["\']([0-9]+)\.([0-9]+)\.([0-9]+)["\']$', re.M)


def main():
    text = INIT_FILE.read_text(encoding="utf-8")
    matches = RE_VERSION.findall(text)
    assert len(matches) == 1, f"Expected one version line in {INIT_FILE}, found {len(matches)}"
    local = tuple(int(part) for part in matches[0])
    published = get_published_version()
    if published is None:
        print(f"{PACKAGE} is not on PyPI yet, keeping version {format_version(local)}")
        return
    if local > published:
        print(
            f"Version {format_version(local)} is higher than {format_version(published)} on "
            f"PyPI, keeping it"
        )
        return
    new = (published[0], published[1], published[2] + 1)
    INIT_FILE.write_text(
        RE_VERSION.sub(f'__version__ = "{format_version(new)}"', text), encoding="utf-8"
    )
    print(
        f"Version {format_version(local)} is not higher than {format_version(published)} on "
        f"PyPI, set it to {format_version(new)}"
    )


def get_published_version() -> tuple[int, int, int] | None:
    """The latest release on PyPI, None if the package was never uploaded."""
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{PACKAGE}/json", timeout=30
        ) as response:
            info = json.load(response)["info"]
    except urllib.error.HTTPError as e:
        # 404 means the package does not exist, everything else is a real problem
        if e.code == 404:
            return None
        raise
    major, minor, patch = info["version"].split(".")
    return int(major), int(minor), int(patch)


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


if __name__ == "__main__":
    main()
