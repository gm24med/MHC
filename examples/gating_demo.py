import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mhc import MHCSequential

def gating_demo():
    print("--- mHC Adaptive Gating Demo ---")

    # 1. Setup a deep network with gating
    dim = 64
    depth = 10

    # Model 1: With Gating (Default)
    model_gated = MHCSequential(
        [nn.Linear(dim, dim)] + [nn.ReLU() for _ in range(depth)],
        use_gating=True
    )

    # Model 2: Without Gating (Just for comparison if needed, currently unused)
    _ = MHCSequential(
        [nn.Linear(dim, dim)] + [nn.ReLU() for _ in range(depth)],
        use_gating=False
    )

    # 2. Simulate training and track gates
    x = torch.randn(16, dim)
    y = torch.randn(16, dim)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model_gated.parameters(), lr=1e-2)

    gate_values = []

    print("Training gated model and tracking gate stabilization...")
    for epoch in range(50):
        optimizer.zero_grad()
        out = model_gated(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Track the gate value of the first skip layer
        with torch.no_grad():
            gate = torch.sigmoid(model_gated.skip_layers[0].gate_logit).item()
            gate_values.append(gate)

    print(f"Final gate value for Layer 0: {gate_values[-1]:.4f}")

    # 3. Plotting the gate evolution
    plt.figure(figsize=(10, 5))
    plt.plot(gate_values, label="Gate Factor (σ(logit))", color="coral", linewidth=2)
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    plt.title("Evolution of mHC Adaptive Gate during Training")
    plt.xlabel("Iteration")
    plt.ylabel("Gate Intensity [0, 1]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig("experiments/results/gating_evolution.png")
    print("Plot saved to experiments/results/gating_evolution.png")

if __name__ == "__main__":
    # Ensure results dir exists
    import os
    os.makedirs("experiments/results", exist_ok=True)
    gating_demo()
