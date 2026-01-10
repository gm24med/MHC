import torch
import pytest
from mhc.constraints import project_simplex, project_identity_preserving

def test_project_simplex_properties():
    logits = torch.randn(2, 4, 5) # (batch, seq, history)
    alphas = project_simplex(logits)
    
    # 1. Non-negative
    assert torch.all(alphas >= 0)
    
    # 2. Sum to 1 across history dim
    sums = alphas.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))

def test_project_identity_preserving_epsilon():
    epsilon = 0.5
    logits = torch.randn(2, 4, 4)
    alphas = project_identity_preserving(logits, epsilon=epsilon)
    
    # 1. Last weight must be at least epsilon
    assert torch.all(alphas[..., -1] >= epsilon)
    
    # 2. Non-negative
    assert torch.all(alphas >= 0)
    
    # 3. Sum to 1
    sums = alphas.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums))

def test_temperature_scaling():
    logits = torch.tensor([1.0, 10.0])
    
    # Low temperature should make it sharper (closer to one-hot)
    alphas_low = project_simplex(logits, temperature=0.1)
    assert alphas_low[1] > 0.99
    
    # High temperature should make it more uniform
    alphas_high = project_simplex(logits, temperature=100.0)
    assert torch.allclose(alphas_high, torch.tensor([0.5, 0.5]), atol=0.1)
