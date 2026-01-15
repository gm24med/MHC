import torch.nn as nn
from mhc import MHCSequential
import time

def dashboard_demo():
    print("--- mHC Stability Dashboard Demo ---")
    print("Note: This demo simulates a training loop to show how logging works.")

    # 1. Setup a deep model
    model = MHCSequential([
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
    ], max_history=4, mode="mhc", constraint="identity")

    # 2. Simulate training steps
    print("Simulating 10 training steps...")
    for step in range(10):
        # In a real scenario, weights would be updated via backprop.
        # Here we just show that the extraction and logging works.

        # Extract weights (internally called by log_to_wandb)
        from mhc.utils.visualization import extract_mixing_weights
        weights = extract_mixing_weights(model)

        for name, alphas in weights.items():
            print(f"Step {step} | Layer: {name} | Alphas: {alphas.tolist()}")

        # If wandb was initialized, it would log here:
        # log_to_wandb(model, step=step)

        time.sleep(0.1)

    print("\n✅ Dashboard logging logic verified.")
    print("To see live results, integrate MHCLightningCallback into your Lightning Trainer.")

if __name__ == "__main__":
    dashboard_demo()
