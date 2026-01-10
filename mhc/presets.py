from typing import Dict

_PRESETS: Dict[str, Dict[str, float]] = {
    "stable": {
        "mode": "mhc",
        "constraint": "identity",
        "epsilon": 0.1,
        "temperature": 1.0,
        "init": "identity",
    },
    "aggressive": {
        "mode": "mhc",
        "constraint": "simplex",
        "epsilon": 0.0,
        "temperature": 0.7,
        "init": "uniform",
    },
    "research": {
        "mode": "mhc",
        "constraint": "simplex",
        "epsilon": 0.05,
        "temperature": 1.0,
        "init": "uniform",
    },
}


def get_preset(name: str) -> Dict[str, float]:
    """Return a copy of a preset config by name."""
    key = name.lower()
    if key not in _PRESETS:
        raise ValueError(f"Unknown preset: {name}")
    return dict(_PRESETS[key])
