"""Conv2D mHC Demo - Computer Vision with Hyper-Connections.

This script demonstrates how to use mHC with convolutional layers
for computer vision tasks.
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from mhc.layers import MHCConv2d, MHCBasicBlock, MHCBottleneck
from mhc import set_seed

logger = logging.getLogger("mhc.examples.conv2d")


def _log(message: str) -> None:
    logger.info(message)


def demo_mhc_conv2d():
    """Demonstrate MHCConv2d layer."""
    _log("\n" + "="*60)
    _log("Demo 1: MHCConv2d Layer")
    _log("="*60)

    # Create a simple conv layer with mHC
    conv = MHCConv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        padding=1,
        max_history=4,
        mode="mhc",
        constraint="identity"
    )

    # Forward pass
    x = torch.randn(8, 3, 32, 32)
    out = conv(x)

    _log(f"Input shape:  {x.shape}")
    _log(f"Output shape: {out.shape}")
    _log(f"History buffer size: {len(conv.history_buffer)}")
    _log("✓ MHCConv2d works like standard Conv2d but with richer skip connections")


def demo_basic_block():
    """Demonstrate MHCBasicBlock."""
    _log("\n" + "="*60)
    _log("Demo 2: MHCBasicBlock (ResNet-style)")
    _log("="*60)

    # Create a ResNet-style basic block
    block = MHCBasicBlock(
        in_channels=64,
        out_channels=64,
        stride=1,
        max_history=4
    )

    x = torch.randn(8, 64, 32, 32)
    out = block(x)

    _log(f"Input shape:  {x.shape}")
    _log(f"Output shape: {out.shape}")
    _log("✓ BasicBlock maintains spatial dimensions")

    # Downsampling block
    downsample_block = MHCBasicBlock(64, 128, stride=2)
    out2 = downsample_block(x)

    _log(f"\nDownsampling block output: {out2.shape}")
    _log("✓ Can downsample with stride=2")


def demo_bottleneck():
    """Demonstrate MHCBottleneck."""
    _log("\n" + "="*60)
    _log("Demo 3: MHCBottleneck (ResNet-50 style)")
    _log("="*60)

    # Create a bottleneck block
    block = MHCBottleneck(
        in_channels=256,
        out_channels=64,
        stride=1
    )

    x = torch.randn(8, 256, 32, 32)
    out = block(x)

    _log(f"Input shape:  {x.shape}")
    _log(f"Output shape: {out.shape} (64 * expansion=4 = 256 channels)")
    _log("✓ Bottleneck uses 1x1 -> 3x3 -> 1x1 convolutions")


def demo_simple_cnn():
    """Demonstrate a simple CNN with mHC."""
    _log("\n" + "="*60)
    _log("Demo 4: Simple CNN with mHC")
    _log("="*60)

    class SimpleMHCCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()

            self.features = nn.Sequential(
                MHCConv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                MHCBasicBlock(64, 64),
                MHCBasicBlock(64, 128, stride=2),
                MHCBasicBlock(128, 128),

                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = SimpleMHCCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)

    _log(f"Input shape:  {x.shape}")
    _log(f"Output shape: {out.shape}")
    _log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    _log("✓ Complete CNN with mHC skip connections")


def demo_training():
    """Demonstrate training with mHC Conv layers."""
    _log("\n" + "="*60)
    _log("Demo 5: Training with mHC (Simulated)")
    _log("="*60)

    set_seed(42)

    # Simple model
    model = nn.Sequential(
        MHCConv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        MHCBasicBlock(32, 32),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10)
    )

    # Dummy data
    X = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train for a few steps
    _log("\nTraining for 5 batches...")
    model.train()

    for i, (batch_x, batch_y) in enumerate(loader):
        if i >= 5:
            break

        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        _log(f"  Batch {i+1}/5, Loss: {loss.item():.4f}")

    _log("\n✓ Training works seamlessly with mHC Conv layers")


def demo_comparison():
    """Compare parameter counts: standard vs mHC."""
    _log("\n" + "="*60)
    _log("Demo 6: Parameter Comparison")
    _log("="*60)

    # Standard ResNet block
    class StandardBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.relu(out + identity)
            return out

    standard = StandardBlock(64)
    mhc_block = MHCBasicBlock(64, 64, max_history=4)

    standard_params = sum(p.numel() for p in standard.parameters())
    mhc_params = sum(p.numel() for p in mhc_block.parameters())

    _log(f"Standard ResNet Block: {standard_params:,} parameters")
    _log(f"mHC BasicBlock:        {mhc_params:,} parameters")
    _log(f"Overhead:              {mhc_params - standard_params} parameters")
    _log(f"Overhead %:            {((mhc_params/standard_params - 1) * 100):.2f}%")
    _log("\n✓ mHC adds minimal parameters (just mixing weights)")


def main():
    """Run all Conv2D demos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _log("\n" + "="*60)
    _log("mHC Conv2D Demo - Computer Vision with Hyper-Connections")
    _log("="*60)

    demo_mhc_conv2d()
    demo_basic_block()
    demo_bottleneck()
    demo_simple_cnn()
    demo_training()
    demo_comparison()

    _log("\n" + "="*60)
    _log("All Conv2D demos complete! ✓")
    _log("="*60)
    _log("\nKey Takeaways:")
    _log("  • MHCConv2d works like standard Conv2d with richer skip connections")
    _log("  • MHCBasicBlock/Bottleneck are drop-in replacements for ResNet blocks")
    _log("  • Training works seamlessly with existing PyTorch code")
    _log("  • Minimal parameter overhead (~0.01% for typical architectures)")
    _log("  • Enables computer vision tasks with mHC benefits")


if __name__ == "__main__":
    main()
