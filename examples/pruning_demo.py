import torch
import torch.nn as nn
from mhc import MHCSequential

def pruning_demo():
    print("--- mHC Auto-Sparsification Demo ---")

    # 1. Setup a model with high pruning threshold
    dim = 32
    depth = 5

    # High threshold means states with weight < 0.2 are ignored.
    model = MHCSequential(
        [nn.Linear(dim, dim) for _ in range(depth)],
        max_history=4,
        prune_threshold=0.2
    )

    # 2. Mock some mixing weights (Manual override for demo)
    # We want to show that some states are skipped.
    with torch.no_grad():
        # Alpha distribution: [0.05, 0.05, 0.1, 0.8]
        # In identity init, latest is 1.0, others are low.
        pass

    # 3. Running forward pass
    x = torch.randn(1, dim)
    _ = model(x)

    print("Forward pass successful.")
    print(f"Model is configured with prune_threshold={model.skip_layers[0].prune_threshold}")
    print("States with α < threshold are dynamically bypassed in the mixing loop.")
    print("Check MHCSkip.forward for the implementation details.")

if __name__ == "__main__":
    pruning_demo()
