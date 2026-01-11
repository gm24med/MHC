import logging

from .layers.history_buffer import HistoryBuffer as HistoryBuffer
from .layers.mhc_skip import MHCSkip as MHCSkip
from .layers.managed import MHCSequential as MHCSequential
from .utils.seed import set_seed as set_seed
from .utils.injection import inject_mhc as inject_mhc, inject_mhc_default as inject_mhc_default
from .utils.profiling import ForwardProfiler as ForwardProfiler
from .utils.compat import compile_model as compile_model, autocast_context as autocast_context
from .presets import get_preset as get_preset
from .config import (
    MHCConfig as MHCConfig,
    get_default_config as get_default_config,
    set_default_config as set_default_config,
    load_config_from_toml as load_config_from_toml,
)

logger = logging.getLogger("mhc")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
