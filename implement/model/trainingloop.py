import torch
import torch.optim as optim
import torch.nn as nn
from samloss import SAMLoss
from Simple3D_SR import HSISuperRes3D
from torch.utils.data import DataLoader, TensorDataset

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize Components
model = HSISuperRes3D(num_bands=103).to(device)
criterion_spatial = nn.MSELoss()
criterion_spectral = SAMLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Dummy Data for Demonstration 
# (Replace this with your 'patches' from the previous step)
# Shape: (Batch, Channel, Bands, H, W)
dummy_input = torch.randn(10, 1, 103, 32, 32).to(device) 
dummy_target = torch.randn(10, 1, 103, 64, 64).to(device)
dataset = TensorDataset(dummy_input, dummy_target)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. Training Loop
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (lr_img, hr_img) in enumerate(loader):
        optimizer.zero_grad()
        
        # Forward Pass
        output = model(lr_img)
        
        # Calculate Hybrid Loss
        # We remove the 'channel' dim for SAM Loss: (B, 1, Bands, H, W) -> (B, Bands, H, W)
        loss_mse = criterion_spatial(output, hr_img)
        loss_sam = criterion_spectral(output.squeeze(1), hr_img.squeeze(1))
        
        # Total Loss (0.1 is a common weight for SAM)
        total_loss = loss_mse + (0.1 * loss_sam)
        
        # Backward Pass
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss/len(loader):.4f}")

print("Training Complete!")