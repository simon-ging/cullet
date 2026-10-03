import importlib.util

import pytest

from cullet import extras


def test_missing_extra_names_the_packages_and_the_install_command(monkeypatch):
    # every other package counts as installed, whatever the environment of the test run has
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: None if name in ("torch", "tqdm") else object()
    )
    with pytest.raises(SystemExit) as error:
        extras.require_full_extra()
    message = str(error.value)
    assert "torch, tqdm" in message and "torchvision" not in message
    assert 'uv tool install "cullet[full]"' in message


def test_installed_extra_passes(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    extras.require_full_extra()
