import torch

class HistoryBuffer:
    """
    A simple buffer to store previous states (tensors) for mixing.
    Manages history length and optional detachment for memory safety or gradient control.
    """
    def __init__(self, max_history: int = 4, detach_history: bool = False):
        self.max_history = max_history
        self.detach_history = detach_history
        self.buffer = []

    def append(self, x: torch.Tensor):
        """
        Append a new state to the buffer.
        """
        if self.detach_history:
            x = x.detach()
        
        self.buffer.append(x)
        
        # Keep only the last max_history items
        if len(self.buffer) > self.max_history:
            self.buffer.pop(0)

    def get(self):
        """
        Returns the current history as a list of tensors.
        """
        return self.buffer

    def clear(self):
        """
        Clears the buffer.
        """
        self.buffer = []

    def __len__(self):
        return len(self.buffer)
