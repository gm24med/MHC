import torch
from mhc.layers.history_buffer import HistoryBuffer

def test_history_buffer_limit():
    buf = HistoryBuffer(max_history=3)
    for i in range(5):
        buf.append(torch.tensor([float(i)]))

    hist = buf.get()
    assert len(hist) == 3
    assert torch.equal(hist[0], torch.tensor([2.0]))
    assert torch.equal(hist[2], torch.tensor([4.0]))

def test_history_buffer_detach():
    buf = HistoryBuffer(max_history=2, detach_history=True)
    x = torch.randn(2, requires_grad=True)
    buf.append(x)

    hist = buf.get()
    assert not hist[0].requires_grad

def test_history_buffer_clear():
    buf = HistoryBuffer(max_history=2)
    buf.append(torch.randn(2))
    buf.clear()
    assert len(buf.get()) == 0
