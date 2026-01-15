# PyTorch Lightning: Industrial Stability Monitoring

Using `mhc` with PyTorch Lightning (PL) transforms your training from a "black box" into a transparent manifold evolution. This guide demonstrates how to leverage the PL ecosystem for deep stability monitoring.

---

## 1. Automated History Management

One of the pain points of custom skip connections is manually clearing history between training batches. In PL, if you don't clear history, gradients might bleed between independent training samples.

The `MHCLightningCallback` handles this automatically:
- **`on_train_batch_start`**: Resets all buffers to ensure samples are independent.
- **`on_validation_batch_start`**: Resets buffers for deterministic validation.
- **`on_test_batch_start`**: Resets buffers for evaluation.

---

## 2. Real-Time Stability Dashboard

By enabling logging in the callback, `mhc` pushes internal manifold statistics to your logger (WandB, TensorBoard, or Comet).

```python
from mhc.utils import MHCLightningCallback
import lightning as L

mhc_callback = MHCLightningCallback(
    log_mixing_weights=True,  # Visualizes heatmaps of history use
    log_entropy=True,         # Measures "specialization" of layers
    log_gradients=True        # Monitors the "History Highway" health
)

trainer = L.Trainer(callbacks=[mhc_callback], ...)
```

### Metrics Explained:

#### A. Mixing Weight Heatmaps
In your dashboard, you will see a heatmap for each layer.
- **Higher weights on index $H-1$**: The model is behaving like a standard ResNet.
- **High weights on index $0$**: The model is intensely reusing features from far back in time.
- **Sparsity**: Zero-weights (black cells) show that the manifold projection has successfully pruned useless paths.

#### B. Shannon Entropy
We calculate the entropy of the mixing vector $\alpha$: $S = -\sum \alpha_i \log \alpha_i$.
- **High Entropy**: The model is unsure and averaging across all history.
- **Low Entropy**: The model has specialized and "locked in" on a specific historical state.
- **Dashboard Trend**: A healthy training run usually shows high entropy early on, which slowly decreases as the model learns its internal routing.

---

## 3. Full Integration Example

```python
class HoneyBadgerModule(L.LightningModule):
    def __init__(self, channels=64, depth=50):
        super().__init__()
        # 1. Define mHC segments
        self.net = MHCSequential(
            [nn.Linear(channels, channels) for _ in range(depth)],
            max_history=4,
            mode="mhc"
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        # MHCSequential handles the history pointers behind the scenes
        preds = self.net(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Start training with the stability watchdog
mhc_watchdog = MHCLightningCallback()
trainer = L.Trainer(callbacks=[mhc_watchdog])
trainer.fit(HoneyBadgerModule(), train_loader)
```

---

## 4. Multi-GPU & Distributed Training (DDP)

`mhc` is fully DDP-compatible.
- **Buffer Synchronization**: History buffers are local to each GPU process. This is correct behavior, as each GPU is processing independent shards of the batch.
- **Logging**: The callback only logs from the global rank 0 by default to avoid duplicate charts in your dashboard.

> [!IMPORTANT]
> When using `strategy="ddp"`, ensure your `MHCSequential` is instantiated inside the `LightningModule.__init__` so that its parameters are correctly registered for gradient synchronization.
