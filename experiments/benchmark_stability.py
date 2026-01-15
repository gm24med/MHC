import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from mhc import MHCSequential

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def create_vanilla_model(depth, dim):
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return x + self.relu(self.linear(x))

def create_resnet_model(depth, dim):
    layers = []
    for _ in range(depth):
        layers.append(ResNetBlock(dim))
    return nn.Sequential(*layers)

def create_mhc_model(depth, dim):
    layers = []
    for _ in range(depth):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.ReLU())
    return MHCSequential(layers, max_history=4, mode="mhc", constraint="identity")

def train_and_collect_stats(model, criterion, optimizer, input_data, target, epochs=100):
    losses = []
    grad_norms = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()

        # Collect gradient norms for the first layer (bottleneck check)
        first_param = next(model.parameters())
        if first_param.grad is not None:
            grad_norms.append(first_param.grad.norm().item())
        else:
            grad_norms.append(0.0)

        optimizer.step()
        losses.append(loss.item())

    return losses, grad_norms

def run_benchmark():
    DEPTH = 50
    DIM = 64
    EPOCHS = 100
    BATCH_SIZE = 32

    print(f"Running benchmark with depth={DEPTH}, dim={DIM}, epochs={EPOCHS}")

    # Synthetic dataset
    X = torch.randn(BATCH_SIZE, DIM)
    Y = torch.randn(BATCH_SIZE, DIM)

    criterion = nn.MSELoss()

    models = {
        "Vanilla": create_vanilla_model(DEPTH, DIM),
        "ResNet": create_resnet_model(DEPTH, DIM),
        "mHC": create_mhc_model(DEPTH, DIM)
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        start_time = time.time()
        losses, grads = train_and_collect_stats(model, criterion, optimizer, X, Y, EPOCHS)
        elapsed = time.time() - start_time
        results[name] = {"losses": losses, "grads": grads, "time": elapsed}
        print(f"Finished {name} in {elapsed:.2f}s")

    # Plotting
    os.makedirs("experiments/results", exist_ok=True)

    # Loss Plot
    plt.figure(figsize=(10, 5))
    for name, data in results.items():
        plt.plot(data["losses"], label=name)
    plt.yscale("log")
    plt.title(f"Convergence Comparison (Depth {DEPTH})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("experiments/results/convergence.png")

    # Gradient Norm Plot
    plt.figure(figsize=(10, 5))
    for name, data in results.items():
        plt.plot(data["grads"], label=name)
    plt.title(f"Gradient Stability - First Layer (Depth {DEPTH})")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("experiments/results/gradients.png")

    print("Benchmark complete. Results saved in experiments/results/")

if __name__ == "__main__":
    run_benchmark()
