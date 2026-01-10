import torch
import numpy as np
from mhc.utils.seed import set_seed
from mhc.utils.tensor_ops import ensure_list, get_last_k

def test_set_seed():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)
    
    np_a = np.random.randn(5)
    set_seed(42)
    np_b = np.random.randn(5)
    assert np.array_equal(np_a, np_b)

def test_ensure_list():
    t = torch.randn(1, 2)
    assert ensure_list(t) == [t]
    assert ensure_list([t]) == [t]

def test_get_last_k():
    hist = [1, 2, 3, 4, 5]
    assert get_last_k(hist, 2) == [4, 5]
    assert get_last_k(hist, 10) == [1, 2, 3, 4, 5]
