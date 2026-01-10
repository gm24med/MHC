import torch
from mhc import MHCSkip

def test_mhc_skip_shapes():
    mhc = MHCSkip(max_history=4)
    x = torch.randn(2, 10, 32)
    history = [torch.randn(2, 10, 32) for _ in range(3)]

    out = mhc(x, history)
    assert out.shape == x.shape

def test_mhc_skip_residual_mode():
    mhc = MHCSkip(mode="residual")
    x = torch.randn(2, 32)
    h = torch.randn(2, 32)
    history = [torch.randn(2, 32), h]

    out = mhc(x, history)
    assert torch.allclose(out, x + h)

def test_mhc_skip_identity_init():
    # In identity init, the latest state should have overwhelming weight
    mhc = MHCSkip(mode="mhc", init="identity", max_history=4)
    x = torch.zeros(2, 32) # Current transform is 0
    h1 = torch.randn(2, 32)
    h_latest = torch.randn(2, 32)
    history = [h1, h_latest]

    out = mhc(x, history)
    # Result should be very close to h_latest
    assert torch.allclose(out, h_latest, atol=1e-3)

def test_mhc_skip_gradient_flow():
    mhc = MHCSkip(mode="mhc", max_history=4)
    x = torch.randn(2, 32, requires_grad=True)
    history = [torch.randn(2, 32, requires_grad=True) for _ in range(2)]

    out = mhc(x, history)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert history[0].grad is not None
    assert mhc.mixing_logits.grad is not None

def test_mhc_skip_no_history():
    mhc = MHCSkip()
    x = torch.randn(2, 32)
    # Should work even with empty history (fallback to identity-like or just returns x)
    # According to our implementation, it returns x if history is empty.
    out = mhc(x, [])
    assert torch.allclose(out, x)
