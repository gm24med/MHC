import torch
from typing import List

class HistoryBuffer:
    """A buffer for storing and managing previous network states for Hyper-Connections.

    This class handles the collection of tensors, maintains a maximum history length,
    and optionally detaches tensors from the computation graph for memory efficiency
    or to prevent unintended backpropagation through history.

    Attributes:
        max_history (int): Maximum number of previous states to store.
        detach_history (bool): If True, tensors are detached before being added to the buffer.
        buffer (List[torch.Tensor]): The internal list storing the historical states.
    """

    def __init__(self, max_history: int = 4, detach_history: bool = False) -> None:
        """Initializes the HistoryBuffer.

        Args:
            max_history: The maximum number of states to keep in history. Defaults to 4.
            detach_history: Whether to detach tensors stored in the buffer. Defaults to False.
        """
        self.max_history = max_history
        self.detach_history = detach_history
        self.buffer: List[torch.Tensor] = []

    def append(self, x: torch.Tensor) -> None:
        """Appends a new state to the history buffer.

        If the buffer exceeds `max_history`, the oldest state is removed.

        Args:
            x: The tensor to add to history.
        """
        if self.detach_history:
            x = x.detach()

        self.buffer.append(x)

        if len(self.buffer) > self.max_history:
            self.buffer.pop(0)

    def get(self) -> List[torch.Tensor]:
        """Retrieves all currently stored states in chronological order.

        Returns:
            List[torch.Tensor]: The list of historical states.
        """
        return list(self.buffer)

    def clear(self) -> None:
        """Removes all states from the buffer."""
        self.buffer = []

    def __len__(self) -> int:
        """Returns the current number of states in the buffer.

        Returns:
            int: Current history length.
        """
        return len(self.buffer)
