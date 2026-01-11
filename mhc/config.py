from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MHCConfig:
    max_history: int = 4
    mode: str = "mhc"
    constraint: str = "simplex"
    epsilon: float = 0.1
    temperature: float = 1.0
    init: str = "identity"
    detach_history: bool = True
    clear_history_each_forward: bool = True
    auto_project: bool = False
    history_scope: str = "module"


_DEFAULT_CONFIG = MHCConfig()
_current_config = _DEFAULT_CONFIG


def get_default_config() -> MHCConfig:
    return _current_config


def set_default_config(config: MHCConfig) -> None:
    global _current_config
    _current_config = config


def resolve_default(value: Any, field: str) -> Any:
    if value is not None:
        return value
    return getattr(get_default_config(), field)


def _load_toml(path: str) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
        loader = tomllib
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore
            loader = tomli
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tomllib not available; install tomli on Python <3.11 to read TOML"
            ) from exc

    with open(path, "rb") as handle:
        return loader.load(handle)


def load_config_from_toml(path: str) -> MHCConfig:
    """Load config from a TOML file under [tool.mhc]."""
    data = _load_toml(path)

    tool_cfg = data.get("tool", {}).get("mhc", {})
    return MHCConfig(**tool_cfg)


def load_config_from_toml_optional(path: str) -> Optional[MHCConfig]:
    try:
        return load_config_from_toml(path)
    except FileNotFoundError:
        return None
