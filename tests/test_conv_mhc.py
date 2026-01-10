"""Tests for Conv2D mHC layers."""

import torch
import torch.nn as nn
import pytest

from mhc.layers.conv_mhc import MHCConv2d, MHCBasicBlock, MHCBottleneck


class TestMHCConv2d:
    """Tests for MHCConv2d layer."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        conv = MHCConv2d(3, 64, kernel_size=3, padding=1, max_history=4)
        x = torch.randn(8, 3, 32, 32)
        out = conv(x)
        
        assert out.shape == (8, 64, 32, 32)
        assert len(conv.history_buffer) == 1
    
    def test_shape_preservation(self):
        """Test that spatial dimensions are preserved with padding."""
        conv = MHCConv2d(16, 32, kernel_size=3, padding=1)
        x = torch.randn(4, 16, 64, 64)
        out = conv(x)
        
        assert out.shape == (4, 32, 64, 64)
    
    def test_stride(self):
        """Test downsampling with stride."""
        conv = MHCConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        x = torch.randn(4, 16, 64, 64)
        out = conv(x)
        
        assert out.shape == (4, 32, 32, 32)
    
    def test_history_accumulation(self):
        """Test that history buffer accumulates states."""
        conv = MHCConv2d(16, 16, kernel_size=3, padding=1, max_history=4)
        x = torch.randn(4, 16, 32, 32)
        
        # Multiple forward passes
        for i in range(5):
            out = conv(x)
        
        # Should have max_history states
        assert len(conv.history_buffer) == 4
    
    def test_clear_history(self):
        """Test clearing history buffer."""
        conv = MHCConv2d(16, 16, kernel_size=3, padding=1)
        x = torch.randn(4, 16, 32, 32)
        
        _ = conv(x)
        assert len(conv.history_buffer) > 0
        
        conv.clear_history()
        assert len(conv.history_buffer) == 0
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        conv = MHCConv2d(16, 32, kernel_size=3, padding=1)
        x = torch.randn(4, 16, 32, 32, requires_grad=True)
        
        out = conv(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert conv.conv.weight.grad is not None
    
    def test_different_modes(self):
        """Test different mHC modes."""
        modes = ["residual", "hc", "mhc"]
        x = torch.randn(4, 16, 32, 32)
        
        for mode in modes:
            conv = MHCConv2d(16, 16, kernel_size=3, padding=1, mode=mode)
            out = conv(x)
            assert out.shape == (4, 16, 32, 32)


class TestMHCBasicBlock:
    """Tests for MHCBasicBlock."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        block = MHCBasicBlock(64, 64, stride=1)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)
        
        assert out.shape == (8, 64, 32, 32)
    
    def test_downsample(self):
        """Test downsampling block."""
        block = MHCBasicBlock(64, 128, stride=2)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)
        
        assert out.shape == (8, 128, 16, 16)
    
    def test_channel_change(self):
        """Test changing number of channels."""
        block = MHCBasicBlock(64, 128, stride=1)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)
        
        assert out.shape == (8, 128, 32, 32)
    
    def test_history_management(self):
        """Test history buffer management."""
        block = MHCBasicBlock(64, 64, max_history=4)
        x = torch.randn(8, 64, 32, 32)
        
        # Multiple forward passes
        for _ in range(5):
            out = block(x)
        
        assert len(block.history_buffer) == 4
    
    def test_clear_history(self):
        """Test clearing history."""
        block = MHCBasicBlock(64, 64)
        x = torch.randn(8, 64, 32, 32)
        
        _ = block(x)
        block.clear_history()
        assert len(block.history_buffer) == 0
    
    def test_gradient_flow(self):
        """Test gradient flow through block."""
        block = MHCBasicBlock(64, 64)
        x = torch.randn(8, 64, 32, 32, requires_grad=True)
        
        out = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert block.conv1.weight.grad is not None
        assert block.conv2.weight.grad is not None


class TestMHCBottleneck:
    """Tests for MHCBottleneck."""
    
    def test_basic_forward(self):
        """Test basic forward pass."""
        block = MHCBottleneck(256, 64, stride=1)
        x = torch.randn(8, 256, 32, 32)
        out = block(x)
        
        # Output should be 64 * expansion = 256
        assert out.shape == (8, 256, 32, 32)
    
    def test_expansion(self):
        """Test channel expansion."""
        block = MHCBottleneck(64, 64, stride=1)
        x = torch.randn(8, 64, 32, 32)
        out = block(x)
        
        # Output should be 64 * 4 = 256
        assert out.shape == (8, 256, 32, 32)
    
    def test_downsample(self):
        """Test downsampling bottleneck."""
        block = MHCBottleneck(256, 64, stride=2)
        x = torch.randn(8, 256, 32, 32)
        out = block(x)
        
        assert out.shape == (8, 256, 16, 16)
    
    def test_history_management(self):
        """Test history buffer."""
        block = MHCBottleneck(256, 64, max_history=4)
        x = torch.randn(8, 256, 32, 32)
        
        for _ in range(5):
            out = block(x)
        
        assert len(block.history_buffer) == 4
    
    def test_gradient_flow(self):
        """Test gradient flow."""
        block = MHCBottleneck(256, 64)
        x = torch.randn(8, 256, 32, 32, requires_grad=True)
        
        out = block(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert block.conv1.weight.grad is not None
        assert block.conv2.weight.grad is not None
        assert block.conv3.weight.grad is not None


class TestConv2DIntegration:
    """Integration tests for Conv2D layers."""
    
    def test_stacked_blocks(self):
        """Test stacking multiple BasicBlocks."""
        blocks = nn.Sequential(
            MHCBasicBlock(64, 64, stride=1),
            MHCBasicBlock(64, 64, stride=1),
            MHCBasicBlock(64, 128, stride=2),
            MHCBasicBlock(128, 128, stride=1),
        )
        
        x = torch.randn(4, 64, 32, 32)
        out = blocks(x)
        
        assert out.shape == (4, 128, 16, 16)
    
    def test_mixed_architecture(self):
        """Test mixing MHCConv2d with regular layers."""
        model = nn.Sequential(
            MHCConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            MHCBasicBlock(64, 64),
            MHCBasicBlock(64, 128, stride=2),
        )
        
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        
        # Should downsample: 224 -> 112 -> 56 -> 28
        assert out.shape == (2, 128, 28, 28)
    
    def test_memory_efficiency(self):
        """Test that detach_history prevents memory accumulation."""
        block = MHCBasicBlock(64, 64, max_history=4)
        x = torch.randn(4, 64, 32, 32, requires_grad=True)
        
        # Run many forward passes
        for _ in range(20):
            out = block(x)
        
        # History should be limited to max_history
        assert len(block.history_buffer) == 4
        
        # History states should be detached
        for state in block.history_buffer.get():
            assert not state.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
