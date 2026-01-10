"""Convolutional layers with mHC skip connections.

This module provides Conv2D layers integrated with Manifold-Constrained
Hyper-Connections, enabling computer vision tasks with richer skip connections.
"""

import torch
import torch.nn as nn

from .mhc_skip import MHCSkip
from .history_buffer import HistoryBuffer


class MHCConv2d(nn.Module):
    """2D Convolutional layer with mHC skip connection.

    This layer wraps a standard Conv2d with an MHCSkip layer, managing
    the history buffer internally. It's designed for easy integration
    into convolutional architectures.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1
        padding: Padding added to input. Default: 0
        dilation: Spacing between kernel elements. Default: 1
        groups: Number of blocked connections. Default: 1
        bias: If True, adds a learnable bias. Default: True
        max_history: Maximum number of historical states to mix. Default: 4
        mode: Mixing mode ('residual', 'hc', 'mhc'). Default: 'mhc'
        constraint: Constraint type for mhc mode. Default: 'simplex'
        epsilon: Minimum weight for identity preservation. Default: 0.1
        temperature: Softmax temperature. Default: 1.0
        detach_history: Whether to detach history tensors. Default: True

    Example:
        >>> conv = MHCConv2d(3, 64, kernel_size=3, padding=1, max_history=4)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([8, 64, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        max_history: int = 4,
        mode: str = "mhc",
        constraint: str = "simplex",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        detach_history: bool = True
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

        self.skip = MHCSkip(
            mode=mode,
            max_history=max_history,
            constraint=constraint,
            epsilon=epsilon,
            temperature=temperature
        )

        self.history_buffer = HistoryBuffer(
            max_history=max_history,
            detach_history=detach_history
        )

        # Projection layer if input/output channels differ
        self.needs_projection = (in_channels != out_channels) or (stride != 1)
        if self.needs_projection:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mHC skip connection.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        # Apply convolution
        out = self.conv(x)

        # Get history
        history = self.history_buffer.get()

        # If we need projection and have history, project historical states
        if self.needs_projection and history:
            history = [self.projection(h) if h.shape != out.shape else h
                      for h in history]

        # Apply mHC skip
        if history:
            out = self.skip(out, history)

        # Update history buffer
        self.history_buffer.append(out)

        return out

    def clear_history(self):
        """Clear the history buffer."""
        self.history_buffer.clear()


class MHCBasicBlock(nn.Module):
    """ResNet-style Basic Block with mHC skip connections.

    This implements the standard ResNet BasicBlock architecture:
    Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> MHCSkip -> ReLU

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution. Default: 1
        max_history: Maximum history for mHC. Default: 4
        mode: Mixing mode. Default: 'mhc'
        constraint: Constraint type. Default: 'simplex'

    Example:
        >>> block = MHCBasicBlock(64, 64, stride=1)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> out = block(x)
        >>> out.shape
        torch.Size([8, 64, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        max_history: int = 4,
        mode: str = "mhc",
        constraint: str = "simplex",
        epsilon: float = 0.1
    ):
        super().__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # mHC skip connection
        self.skip = MHCSkip(
            mode=mode,
            max_history=max_history,
            constraint=constraint,
            epsilon=epsilon
        )

        self.history_buffer = HistoryBuffer(
            max_history=max_history,
            detach_history=True
        )

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the basic block.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Get history and apply mHC skip
        history = self.history_buffer.get()

        # Initialize history with identity if empty
        if not history:
            self.history_buffer.append(identity)
            history = [identity]

        # Apply mHC skip
        out = self.skip(out, history)

        # Update history
        self.history_buffer.append(out)

        # Final activation
        out = self.relu(out)

        return out

    def clear_history(self):
        """Clear the history buffer."""
        self.history_buffer.clear()


class MHCBottleneck(nn.Module):
    """ResNet-style Bottleneck Block with mHC skip connections.

    This implements the ResNet Bottleneck architecture used in ResNet-50/101/152:
    1x1 Conv -> BN -> ReLU -> 3x3 Conv -> BN -> ReLU -> 1x1 Conv -> BN -> MHCSkip -> ReLU

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (before expansion).
        stride: Stride for the 3x3 convolution. Default: 1
        expansion: Channel expansion factor. Default: 4
        max_history: Maximum history for mHC. Default: 4
        mode: Mixing mode. Default: 'mhc'
        constraint: Constraint type. Default: 'simplex'

    Example:
        >>> block = MHCBottleneck(256, 64, stride=1)
        >>> x = torch.randn(8, 256, 32, 32)
        >>> out = block(x)
        >>> out.shape
        torch.Size([8, 256, 32, 32])
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        max_history: int = 4,
        mode: str = "mhc",
        constraint: str = "simplex",
        epsilon: float = 0.1
    ):
        super().__init__()

        expanded_channels = out_channels * self.expansion

        # 1x1 conv (reduce)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv (expand)
        self.conv3 = nn.Conv2d(
            out_channels, expanded_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(expanded_channels)

        self.relu = nn.ReLU(inplace=True)

        # mHC skip connection
        self.skip = MHCSkip(
            mode=mode,
            max_history=max_history,
            constraint=constraint,
            epsilon=epsilon
        )

        self.history_buffer = HistoryBuffer(
            max_history=max_history,
            detach_history=True
        )

        # Downsample if needed
        self.downsample = None
        if stride != 1 or in_channels != expanded_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(expanded_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck block.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out*expansion, H', W').
        """
        identity = x

        # 1x1 reduce
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 expand
        out = self.conv3(out)
        out = self.bn3(out)

        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Get history and apply mHC skip
        history = self.history_buffer.get()

        # Initialize history with identity if empty
        if not history:
            self.history_buffer.append(identity)
            history = [identity]

        # Apply mHC skip
        out = self.skip(out, history)

        # Update history
        self.history_buffer.append(out)

        # Final activation
        out = self.relu(out)

        return out

    def clear_history(self):
        """Clear the history buffer."""
        self.history_buffer.clear()
