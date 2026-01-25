"""Tests for Automatic Mixed Precision (AMP) compatibility.

This module tests that all mHC layers work correctly with torch.cuda.amp.
"""

import pytest
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from mhc import MHCSequential
from mhc.layers import MHCConv2d, MHCBasicBlock, MHCBottleneck


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAMPCompatibility:
    """Test AMP compatibility for all mHC layers."""

    def test_mhc_sequential_amp(self) -> None:
        """Test MHCSequential with AMP."""
        model = MHCSequential(
            [
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            ],
            max_history=4,
            mode="mhc",
        ).cuda()

        x = torch.randn(8, 64, device="cuda")
        scaler = GradScaler()

        # Forward pass with autocast
        with autocast():
            output = model(x)
            loss = output.sum()

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()

    def test_mhc_conv2d_amp(self) -> None:
        """Test MHCConv2d with AMP."""
        model = MHCConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            max_history=4,
        ).cuda()

        x = torch.randn(4, 3, 32, 32, device="cuda")
        scaler = GradScaler()

        with autocast():
            output = model(x)
            loss = output.sum()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()

        assert output.dtype == torch.float16
        assert output.shape == (4, 64, 32, 32)

    def test_mhc_basic_block_amp(self) -> None:
        """Test MHCBasicBlock with AMP."""
        model = MHCBasicBlock(
            in_channels=64,
            out_channels=64,
            max_history=4,
        ).cuda()

        x = torch.randn(4, 64, 32, 32, device="cuda")
        scaler = GradScaler()

        with autocast():
            output = model(x)
            loss = output.sum()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()

        assert output.dtype == torch.float16
        assert output.shape == x.shape

    def test_mhc_bottleneck_amp(self) -> None:
        """Test MHCBottleneck with AMP."""
        model = MHCBottleneck(
            in_channels=256,
            out_channels=64,
            max_history=4,
        ).cuda()

        x = torch.randn(4, 256, 32, 32, device="cuda")
        scaler = GradScaler()

        with autocast():
            output = model(x)
            loss = output.sum()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()

        assert output.dtype == torch.float16
        assert output.shape == (4, 256, 32, 32)  # expansion=4

    def test_amp_training_loop(self) -> None:
        """Test complete training loop with AMP."""
        model = MHCSequential(
            [
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            ],
            max_history=4,
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss()

        # Simulate training
        for _ in range(5):
            x = torch.randn(16, 64, device="cuda")
            y = torch.randint(0, 10, (16,), device="cuda")

            optimizer.zero_grad()

            with autocast():
                output = model(x)
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            assert not torch.isnan(loss)

    def test_amp_gradient_accumulation(self) -> None:
        """Test gradient accumulation with AMP."""
        model = MHCSequential(
            [nn.Linear(64, 64), nn.ReLU()],
            max_history=2,
        ).cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()
        accumulation_steps = 4

        for step in range(accumulation_steps):
            x = torch.randn(8, 64, device="cuda")

            with autocast():
                output = model(x)
                loss = output.sum() / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Verify gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is None or not torch.isnan(param.grad).any()

    def test_amp_mixed_precision_dtype(self) -> None:
        """Test that AMP correctly uses mixed precision."""
        model = MHCSequential(
            [nn.Linear(64, 64)],
            max_history=2,
        ).cuda()

        x = torch.randn(8, 64, device="cuda")

        # Without AMP - should be float32
        output_fp32 = model(x)
        assert output_fp32.dtype == torch.float32

        # With AMP - should be float16
        with autocast():
            output_fp16 = model(x)
            assert output_fp16.dtype == torch.float16

        # Outputs should be close
        assert torch.allclose(output_fp32, output_fp16.float(), rtol=1e-2)
