"""Visualization Demo for mHC.

This script demonstrates all visualization capabilities of the mHC library,
including mixing weights heatmaps, gradient flow analysis, and training dashboards.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from mhc import MHCSequential, set_seed
from mhc.utils.visualization import (
    plot_mixing_weights,
    plot_gradient_flow,
    plot_history_contribution,
    create_training_dashboard,
    extract_mixing_weights
)


def create_sample_model(input_dim=64, hidden_dim=64, num_layers=5):
    """Create a sample MHC model for demonstration."""
    layers = []
    for i in range(num_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        ])
    
    model = MHCSequential(
        layers,
        max_history=4,
        mode="mhc",
        constraint="identity",
        epsilon=0.1
    )
    
    return model


def demo_mixing_weights_plot():
    """Demonstrate mixing weights visualization."""
    print("\n" + "="*60)
    print("Demo 1: Mixing Weights Heatmap")
    print("="*60)
    
    set_seed(42)
    model = create_sample_model()
    
    # Run a forward pass to initialize
    x = torch.randn(8, 64)
    _ = model(x)
    
    # Plot mixing weights
    fig = plot_mixing_weights(model, save_path="mixing_weights.png")
    print("✓ Mixing weights heatmap saved to 'mixing_weights.png'")
    print("  This shows how each layer weights different historical states")
    
    # Extract and print weights
    weights = extract_mixing_weights(model)
    print(f"\n  Found {len(weights)} MHCSkip layers")
    for name, alphas in list(weights.items())[:2]:  # Show first 2
        print(f"  {name}: {alphas.numpy()}")


def demo_gradient_flow_plot():
    """Demonstrate gradient flow visualization."""
    print("\n" + "="*60)
    print("Demo 2: Gradient Flow Analysis")
    print("="*60)
    
    set_seed(42)
    model = create_sample_model()
    
    # Create dummy data
    x = torch.randn(8, 64)
    y = torch.randn(8, 64)
    
    # Plot gradient flow
    fig = plot_gradient_flow(
        model, 
        x, 
        target=y,
        loss_fn=nn.MSELoss(),
        save_path="gradient_flow.png"
    )
    print("✓ Gradient flow plot saved to 'gradient_flow.png'")
    print("  This shows gradient magnitudes across all layers")
    print("  Helps identify vanishing/exploding gradient issues")


def demo_history_contribution_plot():
    """Demonstrate history contribution visualization."""
    print("\n" + "="*60)
    print("Demo 3: History Contribution for Single Layer")
    print("="*60)
    
    set_seed(42)
    model = create_sample_model()
    
    # Run forward pass
    x = torch.randn(8, 64)
    _ = model(x)
    
    # Get mixing weights from first skip layer
    weights = extract_mixing_weights(model)
    first_layer_name = list(weights.keys())[0]
    first_layer_weights = weights[first_layer_name]
    
    # Plot
    fig = plot_history_contribution(
        first_layer_weights,
        layer_name=first_layer_name,
        save_path="history_contribution.png"
    )
    print(f"✓ History contribution plot saved to 'history_contribution.png'")
    print(f"  Layer: {first_layer_name}")
    print(f"  Weights: {first_layer_weights.numpy()}")
    print("  Shows how much each historical state contributes")


def demo_training_dashboard():
    """Demonstrate training dashboard with simulated training."""
    print("\n" + "="*60)
    print("Demo 4: Training Dashboard (Simulated Training)")
    print("="*60)
    
    set_seed(42)
    model = create_sample_model()
    
    # Create dummy dataset
    X = torch.randn(100, 64)
    y = torch.randn(100, 64)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Track metrics
    metrics = {
        'loss': [],
        'mixing_weights_history': []
    }
    
    # Simulate training
    num_epochs = 10
    print(f"\n  Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        metrics['loss'].append(avg_loss)
        
        # Track mixing weights from first layer
        weights = extract_mixing_weights(model)
        if weights:
            first_weights = list(weights.values())[0].numpy()
            metrics['mixing_weights_history'].append(first_weights)
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Create dashboard
    fig = create_training_dashboard(
        metrics,
        save_path="training_dashboard.png"
    )
    print("\n✓ Training dashboard saved to 'training_dashboard.png'")
    print("  Shows loss curve and mixing weights evolution over training")


def main():
    """Run all visualization demos."""
    print("\n" + "="*60)
    print("mHC Visualization Tools Demo")
    print("="*60)
    print("\nThis demo will create 4 visualization plots:")
    print("  1. mixing_weights.png - Heatmap of mixing weights")
    print("  2. gradient_flow.png - Gradient magnitudes across layers")
    print("  3. history_contribution.png - Single layer contribution")
    print("  4. training_dashboard.png - Training metrics dashboard")
    
    # Run all demos
    demo_mixing_weights_plot()
    demo_gradient_flow_plot()
    demo_history_contribution_plot()
    demo_training_dashboard()
    
    print("\n" + "="*60)
    print("All visualizations complete! ✓")
    print("="*60)
    print("\nCheck the current directory for the generated plots.")
    print("These visualizations help you understand:")
    print("  • How mHC mixes historical states")
    print("  • Whether gradients flow properly")
    print("  • How mixing patterns evolve during training")


if __name__ == "__main__":
    main()
