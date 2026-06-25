"""
Demo script to show the progress bars in action
Run this to see a sample of how the training progress bars work
"""

import sys
sys.path.insert(0, r"e:\Research\svnit isro\implement\model")

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import time

# Create a sample model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 100)
    
    def forward(self, x):
        return self.linear(x)

# Demo training with progress bars
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\n" + "="*70)
print("🎓 DEMO: Training Progress Bars")
print("="*70)

epochs = 5
batch_per_epoch = 4

# Create progress bar for epochs
epoch_pbar = tqdm(total=epochs, desc="Training Progress", unit="epoch")

for epoch in range(epochs):
    epoch_loss = 0
    
    # Progress bar for batches within epoch
    batch_pbar = tqdm(total=batch_per_epoch, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
    
    for batch in range(batch_per_epoch):
        # Simulate training
        x = torch.randn(16, 100).to(device)
        y = torch.randn(16, 100).to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_pbar.update(1)
        batch_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        time.sleep(0.2)  # Simulate computation
    
    batch_pbar.close()
    
    avg_loss = epoch_loss / batch_per_epoch
    epoch_pbar.update(1)
    epoch_pbar.set_postfix({'loss': f'{avg_loss:.6f}'}, refresh=True)

epoch_pbar.close()

print("\n" + "="*70)
print("✅ Demo Complete!")
print("="*70)
print("\n📊 Key Features:")
print("  ✓ Overall epoch progress bar (0-100%)")
print("  ✓ Per-batch progress within each epoch")
print("  ✓ Real-time loss display")
print("  ✓ Completion time tracking")
print("\nYour actual training will show similar progress indicators!")
print("="*70 + "\n")
