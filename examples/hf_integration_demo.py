import torch
from transformers import ViTModel, ViTConfig
from mhc import inject_mhc

def vit_injection_demo():
    print("--- Hugging Face ViT Injection Demo ---")

    # 1. Load a standard ViT (or BERT, etc.)
    config = ViTConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=2)
    model = ViTModel(config)

    print(f"Original model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Inject mHC into the attention layers (or FeedForward layers)
    # Target common HF module names like 'ViTSelfAttention' or just generic 'Linear'
    from torch import nn

    # We target nn.Linear within the architecture to add richer skips to hidden paths
    inject_mhc(
        model,
        target_types=[nn.Linear],
        max_history=4,
        mode="mhc",
        constraint="identity"
    )

    print(f"Injected model parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Running a forward pass
    x = torch.randn(1, 197, 64) # (batch, seq, hidden)
    output = model(x)

    print(f"Forward pass successful. Output shape: {output.last_hidden_state.shape}")
    print("mHC history buffer is automatically managed via forward pre-hooks.")

if __name__ == "__main__":
    try:
        vit_injection_demo()
    except ImportError:
        print("Please install 'transformers' to run this demo.")
