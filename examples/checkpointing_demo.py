import torch
import torch.nn as nn
from mhc import MHCSequential

def checkpointing_demo():
    print("--- mHC Gradient Checkpointing Demo ---")

    # 1. Setup a very deep model to test memory/gradients
    dim = 128
    depth = 50 # deep enough to benefit from checkpointing

    modules = [nn.Linear(dim, dim) for _ in range(depth)]

    # Model with checkpointing
    model = MHCSequential(
        modules,
        max_history=4,
        use_checkpointing=True
    )

    print(f"Model depth: {depth} layers")
    print(f"Checkpointing enabled: {model.use_checkpointing}")

    # 2. Verify backward pass
    x = torch.randn(8, dim, requires_grad=True)
    out = model(x)
    loss = out.mean()

    print("Computing backward pass...")
    loss.backward()

    # Check if gradients flowed to the input
    if x.grad is not None:
        print("✅ Gradient flow verified (x.grad exists).")
    else:
        print("❌ Gradient flow failed!")

    # Check if gradients exist in the model parameters
    has_grads = all(p.grad is not None for p in model.parameters())
    if has_grads:
        print("✅ All model parameters have gradients.")
    else:
        print("❌ Some model parameters are missing gradients.")

    print("\nMemory check: Gradient checkpointing allows training deeper models")
    print("by recomputing activations during the backward pass.")

if __name__ == "__main__":
    checkpointing_demo()
