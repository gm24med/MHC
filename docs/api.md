# API Reference

This page provides an overview of the main classes and functions in the `mhc` package.

::: mhc.layers.mhc_skip.MHCSkip
    selection:
      members:
        - __init__
        - forward

::: mhc.layers.managed.MHCSequential
    selection:
      members:
        - __init__
        - forward

::: mhc.layers.history_buffer.HistoryBuffer
    selection:
      members:
        - __init__
        - append
        - get
        - clear

::: mhc.utils.injection.inject_mhc
::: mhc.utils.seed.set_seed
::: mhc.utils.logging.get_logger
::: mhc.config.MHCConfig
::: mhc.config.get_default_config
::: mhc.config.set_default_config
::: mhc.config.load_config_from_toml

## TensorFlow (optional)

::: mhc.tf.TFMHCSkip
::: mhc.tf.TFMatrixMHCSkip
::: mhc.tf.TFMHCSequential
::: mhc.tf.TFMHCSequentialGraph
::: mhc.tf.TFHistoryBufferGraph
