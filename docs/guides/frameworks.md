# 🛠️ Framework Integrations

mHC is designed to play well with the modern ML ecosystem.

## PyTorch Lightning

Use the `MHCLightningCallback` to automatically clear history and log stability metrics.

```python
from mhc.utils.lightning_callback import MHCLightningCallback
import lightning as L

model = MyLightningModel() # Using MHCSequential inside
trainer = L.Trainer(
    callbacks=[MHCLightningCallback(log_mixing_weights=True)]
)

trainer.fit(model, train_loader)
```

## Hugging Face Transformers

Inject mHC into Transformers models to boost performance or stability during fine-tuning.

```python
from transformers import AutoModel
from mhc import inject_mhc
import torch.nn as nn

model = AutoModel.from_pretrained("bert-base-uncased")

# Inject into all Linear layers (Attention & MLP)
inject_mhc(model, target_types=[nn.Linear], max_history=6)

# The model now has history-aware skip connections!
```

## Custom Modules (`@mhc_compatible`)

If you are building a custom research module, use the decorator to signal compatibility.

```python
from mhc.utils.decorators import mhc_compatible
import torch.nn as nn

@mhc_compatible
class MyCustomCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x):
        return self.net(x)
```

---

### Best Practices
- **History Size**: `max_history=4` or `max_history=6` is usually sufficient.
- **Constraints**: Use `constraint="identity"` for maximum training stability.
- **Detaching**: Set `detach_history=True` in `MHCSequential` if your network is extremely deep (100+ instances of skips) to save memory.
