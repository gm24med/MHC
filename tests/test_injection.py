import torch
import torch.nn as nn
from mhc import inject_mhc, inject_mhc_default

def test_inject_mhc_linear():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10)
    )

    # Inject mHC into Linears
    inject_mhc(model, target_types=nn.Linear, max_history=2)

    # Verify structure (wrapped in InjectedMHC)
    assert "InjectedMHC" in str(model[0].__class__)
    assert "InjectedMHC" in str(model[2].__class__)

    # Test forward
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 10)

def test_recursive_injection():
    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(5, 5))
            self.head = nn.Linear(5, 2)

    model = DeepModel()
    inject_mhc(model, target_types=nn.Linear)

    # Check that both nested and top-level linears are wrapped
    assert "InjectedMHC" in str(model.net[0].__class__)
    assert "InjectedMHC" in str(model.head.__class__)

def test_inject_mhc_default_common_layers():
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.LayerNorm(8),
        nn.Linear(8, 4),
    )

    inject_mhc_default(model, max_history=2)

    assert "InjectedMHC" in str(model[0].__class__)
    assert "InjectedMHC" in str(model[1].__class__)
    assert "InjectedMHC" in str(model[2].__class__)


def test_inject_mhc_history_scope_module():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    inject_mhc(model, target_types=nn.Linear, history_scope="module")

    first = model[0]
    second = model[2]

    assert first.history_buffer is not second.history_buffer

def test_injection_shape_change_resets_history():
    model = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
    )
    inject_mhc(model, target_types=nn.Linear, max_history=3)

    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 8)
