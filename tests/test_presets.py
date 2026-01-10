import pytest

from mhc import get_preset


def test_get_preset_valid():
    preset = get_preset("stable")
    assert preset["mode"] == "mhc"
    assert "temperature" in preset


def test_get_preset_invalid():
    with pytest.raises(ValueError):
        get_preset("unknown")
