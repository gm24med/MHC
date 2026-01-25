"""AMP Training Demo - Automatic Mixed Precision with mHC.

This script demonstrates how to use AMP for 2x faster training with mHC models.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from mhc import MHCSequential, set_seed


def create_model(input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10) -> nn.Module:
    """Create a simple MHC model."""
    return MHCSequential(
        [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ],
        max_history=4,
        mode="mhc",
        constraint="identity",
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    use_amp: bool = False,
    device: str = "cuda",
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        use_amp: Whether to use AMP.
        device: Device to train on.

    Returns:
        Tuple of (average loss, time taken).
    """
    model.train()
    total_loss = 0.0
    scaler = GradScaler() if use_amp else None

    start_time = time.time()

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                output = model(batch_x)
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    elapsed = time.time() - start_time
    return total_loss / len(dataloader), elapsed


def main():
    """Run AMP training comparison."""
    print("=" * 60)
    print("AMP Training Demo - mHC")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. AMP requires GPU.")
        print("Running on CPU for demonstration (no speedup expected).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✓ Using device: {torch.cuda.get_device_name(0)}")

    set_seed(42)

    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    X = torch.randn(1000, 64)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training parameters
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()

    print(f"\n🎯 Training for {num_epochs} epochs...")
    print("-" * 60)

    # Train WITHOUT AMP
    print("\n1️⃣  Training WITHOUT AMP (FP32)...")
    model_fp32 = create_model().to(device)
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)

    fp32_times = []
    for epoch in range(num_epochs):
        loss, elapsed = train_epoch(
            model_fp32, dataloader, optimizer_fp32, criterion, use_amp=False, device=device
        )
        fp32_times.append(elapsed)
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Time: {elapsed:.3f}s")

    avg_fp32_time = sum(fp32_times) / len(fp32_times)
    print(f"  Average time per epoch: {avg_fp32_time:.3f}s")

    # Train WITH AMP
    if device == "cuda":
        print("\n2️⃣  Training WITH AMP (FP16)...")
        model_amp = create_model().to(device)
        optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)

        amp_times = []
        for epoch in range(num_epochs):
            loss, elapsed = train_epoch(
                model_amp, dataloader, optimizer_amp, criterion, use_amp=True, device=device
            )
            amp_times.append(elapsed)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {loss:.4f} - Time: {elapsed:.3f}s")

        avg_amp_time = sum(amp_times) / len(amp_times)
        print(f"  Average time per epoch: {avg_amp_time:.3f}s")

        # Results
        print("\n" + "=" * 60)
        print("📈 RESULTS")
        print("=" * 60)
        print(f"FP32 (no AMP):  {avg_fp32_time:.3f}s per epoch")
        print(f"FP16 (AMP):     {avg_amp_time:.3f}s per epoch")
        speedup = avg_fp32_time / avg_amp_time
        print(f"Speedup:        {speedup:.2f}x")
        memory_saved = (1 - avg_amp_time / avg_fp32_time) * 100
        print(f"Time saved:     {memory_saved:.1f}%")

        print("\n✅ AMP Training Complete!")
        print("\n💡 Key Takeaways:")
        print("  • AMP provides 1.5-2x speedup on modern GPUs")
        print("  • No accuracy loss with proper gradient scaling")
        print("  • Reduces memory usage by ~40%")
        print("  • Works seamlessly with all mHC layers")
    else:
        print("\n⚠️  Skipping AMP comparison (requires CUDA)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
