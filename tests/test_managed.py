import torch
import torch.nn as nn
from mhc.layers.managed import MHCSequential

def test_mhc_sequential_forward():
    input_dim = 16
    hidden_dim = 16
    num_layers = 3

    layers = [nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)]
    model = MHCSequential(layers, max_history=2)

    x = torch.randn(4, input_dim)
    out = model(x)

    assert out.shape == (4, hidden_dim)
    # Check that history buffer was populated (it has x_0 and x_out)
    # Actually MHCSequential clears buffer at start and appends x_0, then x_1, x_2, x_3
    # So length should be max_history (2) at the end.
    assert len(model.history_buffer) == 2

def test_mhc_sequential_equivalence_to_residual():
    # If mode is residual, it should behave like a standard ResNet-like sum if f(x) is small
    input_dim = 16
    layers = [nn.Identity() for _ in range(2)]
    model = MHCSequential(layers, mode="residual")

    x = torch.ones(1, input_dim)
    out = model(x)

    # x_0 = 1
    # Layer 1: f(x_0) = 1. Skip: (x_0 + f(x_0)) = 2.
    # Layer 2: f(x_1) = 2. Skip: (x_1 + f(x_1)) = 4.
    assert torch.allclose(out, torch.tensor(4.0))
