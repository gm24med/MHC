# Configuration System API

mHC includes a centralized configuration system that allows you to manage model hyperparameters globally, ensuring consistency across deep nested architectures either via Python code or through a standard `pyproject.toml` file.

---

## 1. The Singleton Configuration Pattern

While every individual layer can be configured via its constructor, managing 100+ identical layers manually is error-prone. `mhc` uses a singleton configuration pattern to maintain global state.

### Setting Defaults in Python

This is the recommended way to enforce architectural consistency across your entire experiment.

```python
import mhc

# 1. Update the global singleton state
mhc.set_default_config(
    max_history=6,
    mode="mhc",
    constraint="identity",
    epsilon=0.2,
    detach_history=True
)

# 2. Subsequent layers will inherit these settings automatically
# even if no arguments are passed.
model = mhc.MHCSequential(layers)
```

---

## 2. Declarative Configuration (`pyproject.toml`)

mHC is "Environment Aware." It can automatically read the `tool.mhc` section from your project's configuration file, allowing you to change architecture settings without touching a single line of Python code.

### Example `pyproject.toml`:

```toml
[tool.mhc]
max_history = 8
mode = "mhc"
constraint = "simplex"
detach_history = true
temperature = 0.8
auto_project = true
```

### Loading the File:

```python
from mhc import load_config_from_toml, set_default_config

# 1. Parse the TOML file
config = load_config_from_toml("pyproject.toml")

# 2. Inject it into the mhc engine
set_default_config(config)
```

---

## 3. Data Field Reference

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_history` | `int` | `4` | The size of the history sliding window. |
| `mode` | `str` | `"mhc"` | Mixing mode: `"mhc"`, `"hc"`, or `"residual"`. |
| `constraint` | `str` | `"simplex"`| Projected manifold: `"simplex"` or `"identity"`. |
| `epsilon` | `float` | `0.1` | The stability floor for the identity constraint. |
| `temperature` | `float` | `1.0` | Scalar for logit sharpness before projection. |
| `detach_history`| `bool` | `False` | Prevents gradients from flowing into the history window. |
| `auto_project` | `bool` | `False` | Automatically resolves shape/channel mismatches. |

---

## 4. Hierarchy of Precedence

mHC resolves configurations using a strict hierarchy to avoid ambiguity. If a parameter is defined in multiple places, the library follows this order of importance (highest to lowest):

1.  **Explicit Constructor Argument**: (e.g., `MHCSequential(max_history=10)`)
2.  **Explicit Config Object**: Passing an `MHCConfig` instance to the constructor.
3.  **Local Project Config**: Settings loaded from `pyproject.toml`.
4.  **Global Singleton**: Settings defined via `mhc.set_default_config()`.
5.  **Library Defaults**: The internal "Honey Badger" defaults.

---

## 5. The Core `MHCConfig` Class

For advanced users and library developers, we expose the underlying `dataclass` used for state management. This is useful for passing around groups of settings without affecting the global singleton.

```python
from mhc.config import MHCConfig

# Create a local, immutable config pack
research_config = MHCConfig(
    max_history=16,
    mode="mhc",
    auto_project=True,
    stochastic=True
)

# Use it specifically for one branch of your model
deep_branch = MHCSequential(layers, config=research_config)
```
