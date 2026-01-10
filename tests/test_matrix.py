import torch
from mhc.constraints.matrix import project_doubly_stochastic
from mhc.layers.matrix_skip import MatrixMHCSkip

def test_project_doubly_stochastic():
    logits = torch.randn(4, 4)
    M = project_doubly_stochastic(logits, iterations=20)
    
    # Check row sums
    assert torch.allclose(M.sum(dim=-1), torch.ones(4), atol=1e-3)
    # Check col sums
    assert torch.allclose(M.sum(dim=-2), torch.ones(4), atol=1e-3)

def test_matrix_mhc_skip_forward():
    mhc = MatrixMHCSkip(max_history=3)
    x = torch.randn(2, 10)
    history = [torch.randn(2, 10) for _ in range(3)]
    
    out = mhc(x, history)
    assert out.shape == (2, 10)
