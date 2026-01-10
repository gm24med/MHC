import torch
import torch.nn as nn

from mhc.utils.profiling import ForwardProfiler


def test_forward_profiler_collects_timings():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    profiler = ForwardProfiler(model, filter_fn=lambda name, module: name != "")
    profiler.start()

    x = torch.randn(2, 4)
    _ = model(x)

    profiler.stop()
    summary = profiler.summary()
    assert summary
    assert all(value >= 0.0 for value in summary.values())
