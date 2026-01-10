import torch
import torch.nn as nn
from typing import List
from ..constraints.matrix import project_doubly_stochastic

class MatrixMHCSkip(nn.Module):
    """Advanced Hyper-Connections layer that uses Matrix Mixing.
    
    Instead of a single scalar per history element, this layer uses a 
    learnable mixing matrix. It can be constrained to be doubly stochastic.
    
    Attributes:
        max_history (int): Max history window.
        mixing_matrix (nn.Parameter): Learnable (K, K) or (K, D) matrix.
    """
    def __init__(
        self,
        max_history: int = 4,
        doubly_stochastic: bool = True,
        iterations: int = 10
    ):
        super().__init__()
        self.max_history = max_history
        self.doubly_stochastic = doubly_stochastic
        self.iterations = iterations
        
        # Simple implementation: learn a matrix of size (max_history, max_history)
        # This assumes history tensors are stacked.
        self.mixing_logits = nn.Parameter(torch.randn(max_history, max_history) * 0.01)

    def forward(self, x: torch.Tensor, history: List[torch.Tensor]) -> torch.Tensor:
        if not history:
            return x
            
        hist_window = history[-self.max_history:]
        K = len(hist_window)
        
        # Stack history into (Batch, K, Dim)
        # Assuming last dim is features
        # For simplicity, we assume history tensors have same shape
        H = torch.stack(hist_window, dim=1) # (B, K, ...)
        
        logits = self.mixing_logits[-K:, -K:]
        
        if self.doubly_stochastic:
            weights = project_doubly_stochastic(logits, iterations=self.iterations)
        else:
            weights = torch.softmax(logits, dim=-1)
            
        # history_mix = Weights @ History
        # We need to broadcast weights correctly
        # Weights: (K, K), History: (B, K, D)
        # Result: (B, K, D) -> But we usually only want one output vector to add to x.
        # Actually, matrix mixing often implies mixing ACROSS states to produce new states.
        # For a standard skip, we'll just take the "most recent" combined output.
        
        mixed_history = torch.matmul(weights, H.flatten(2)) # (B, K, D_flat)
        mixed_history = mixed_history.view_as(H)
        
        # Take the last mixed state as the skip connection
        return x + mixed_history[:, -1]
