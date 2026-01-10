import torch
import torch.nn as nn
import torch.optim as optim
from mhc import MHCSkip, HistoryBuffer, set_seed
import argparse
import time

class DeepMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=20, mode="mhc", constraint="simplex", max_history=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.skips = nn.ModuleList([MHCSkip(mode=mode, constraint=constraint, max_history=max_history) for _ in range(num_layers)])
        self.buffers = [HistoryBuffer(max_history=max_history, detach_history=True) for _ in range(num_layers)]
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
            
            # Mix with history
            # In this simple setup, we use a separate buffer per layer for simplicity
            # OR one could use a single global history. 
            # For a stress test, layer-wise history is easier to track.
            hist = self.buffers[i].get()
            x = self.skips[i](x, hist)
            self.buffers[i].append(x)
        return x

def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} | Mode: {args.tag} | Layers: {args.layers}")

    model = DeepMLP(
        num_layers=args.layers, 
        mode=args.mode, 
        constraint=args.constraint, 
        max_history=args.max_history
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Dummy data
    x_train = torch.randn(128, 64).to(device)
    y_train = torch.randn(128, 64).to(device)

    start_time = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        
        if torch.isnan(loss):
            print(f"Diverged at step {step} (NaN loss)")
            return {"status": "diverged", "step": step}
            
        loss.backward()
        
        # Log gradient norm of the first layer to check for vanishing/exploding gradients
        first_layer_grad_norm = model.layers[0].weight.grad.norm().item()
        
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f} | GradNorm (L0): {first_layer_grad_norm:.6f}")

    end_time = time.time()
    print(f"Finished {args.steps} steps in {end_time - start_time:.2f}s")
    return {"status": "success", "loss": loss.item()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="mhc", choices=["residual", "hc", "mhc"])
    parser.add_argument("--constraint", type=str, default="simplex", choices=["simplex", "identity"])
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--max_history", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="mhc-simplex")
    
    args = parser.parse_args()
    run_experiment(args)
