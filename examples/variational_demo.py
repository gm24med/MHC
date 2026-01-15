import torch
import torch.nn as nn
from mhc import MHCSequential

def variational_demo():
    print("--- mHC Variational (Stochastic) Mixing Demo ---")

    # 1. Setup a model with stochastic mixing
    dim = 32
    depth = 3

    model = MHCSequential(
        [nn.Linear(dim, dim) for _ in range(depth)],
        max_history=4,
        stochastic=True,
        temperature=0.5 # Lower temperature = sharper (more categorical) samples
    )

    # 2. Verify stochasticity in training mode
    model.train()
    x = torch.randn(1, dim)

    print("Running multiple forward passes in training mode...")
    outputs = []
    for i in range(5):
        out = model(x)
        outputs.append(out.clone().detach())
        print(f"Pass {i} | Output mean: {out.mean().item():.6f}")

    # Check if outputs are different (due to stochastic mixing)
    is_stochastic = any(not torch.allclose(outputs[0], out) for out in outputs[1:])
    if is_stochastic:
        print("✅ Stochasticity verified: Outputs vary between passes.")
    else:
        # Note: with very high/low weights it might be close,
        # but in general with 50/50 it should be different.
        print("ℹ️ Outputs are identical. (Possibly fixed init or sharp weights).")

    # 3. Verify determinism in eval mode
    model.eval()
    print("\nRunning in eval mode (deterministic)...")
    out1 = model(x)
    out2 = model(x)

    if torch.allclose(out1, out2):
        print("✅ Determinism verified: Outputs are identical in eval mode.")
    else:
        print("❌ Determinism check failed in eval mode!")

if __name__ == "__main__":
    variational_demo()
