import importlib.util

import pytest

from cullet import extras


def test_missing_extra_names_the_packages_and_the_install_command(monkeypatch):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name in ("torch", "tqdm") else real_find_spec(name),
    )
    with pytest.raises(SystemExit) as error:
        extras.require_dedup_extra()
    message = str(error.value)
    assert "torch, tqdm" in message and "torchvision" not in message
    assert 'uv tool install "cullet[dedup]"' in message


def test_installed_extra_passes(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    extras.require_dedup_extra()
