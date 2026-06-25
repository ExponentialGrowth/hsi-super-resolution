import os
import argparse
import time
import json
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# ==========================================
# 1. 3D MODEL ARCHITECTURES
# ==========================================

class HSISuperRes3D(nn.Module):
    def __init__(self, in_channels=1, num_bands=103):
        super(HSISuperRes3D, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True)
        )
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.reconstruction = nn.Conv3d(64, in_channels, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        residual = self.upsampler(x) 
        out = self.feature_extractor(x)
        out = self.upsampler(out)
        out = self.reconstruction(out)
        return out + residual

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  
        return self.relu(out)

class D3CNN(nn.Module):
    def __init__(self, num_blocks=5):
        super(D3CNN, self).__init__()
        self.head = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(*[ResidualBlock3D(64) for _ in range(num_blocks)])
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.tail = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res  
        x = self.upsampler(x)
        return self.tail(x)

class Lightweight3DCNN(nn.Module):
    def __init__(self, bands=103):
        super(Lightweight3DCNN, self).__init__()
        self.head = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.tail = nn.Conv3d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.head(x))
        res = self.body(x)
        x = x + res
        x = self.upsampler(x)
        return self.tail(x)

class Deep3DResNet(nn.Module):
    def __init__(self, bands=103, num_blocks=8):
        super(Deep3DResNet, self).__init__()
        self.head = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock3D(64) for _ in range(num_blocks)])
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.tail = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.head(x))
        res = self.body(x)
        x = x + res
        x = self.upsampler(x)
        return self.tail(x)

class Dense3DNet(nn.Module):
    def __init__(self, bands=103):
        super(Dense3DNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.tail = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        x2 = x2 + x1  
        x3 = torch.relu(self.conv3(x2))
        x3 = x3 + x2  
        x4 = torch.relu(self.conv4(x3))
        x4 = x4 + x3  
        x5 = torch.relu(self.conv5(x4))
        x5 = x5 + x4  
        x = self.upsampler(x5)
        return self.tail(x)

class Hybrid3DNet(nn.Module):
    def __init__(self, bands=103):
        super(Hybrid3DNet, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )
        self.spectral = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.tail = nn.Conv3d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        spatial_feat = self.spatial(x)
        spectral_feat = self.spectral(x)
        fused = torch.cat([spatial_feat, spectral_feat], dim=1)
        fused = torch.relu(self.fusion(fused))
        fused = self.upsampler(fused)
        return self.tail(fused)

# ==========================================
# 2. LOSS FUNCTIONS
# ==========================================

class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, img1, img2):
        # Flatten spatial dimensions to (Batch, Bands, Pixels)
        img1 = img1.view(img1.shape[0], img1.shape[1], -1)
        img2 = img2.view(img2.shape[0], img2.shape[1], -1)

        dot_product = torch.sum(img1 * img2, dim=1)
        img1_norm = torch.sqrt(torch.sum(img1**2, dim=1))
        img2_norm = torch.sqrt(torch.sum(img2**2, dim=1))

        cos_theta = dot_product / (img1_norm * img2_norm + 1e-8)
        sam_angles = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        return torch.mean(sam_angles)

class HybridLoss(nn.Module):
    def __init__(self, sam_weight=0.1):
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.sam = SAMLoss()
        self.sam_weight = sam_weight

    def forward(self, out, target):
        loss_mse = self.mse(out, target)
        loss_sam = self.sam(out.squeeze(1), target.squeeze(1))
        return loss_mse + (self.sam_weight * loss_sam)

# ==========================================
# 3. DATASET & PRE-PROCESSING
# ==========================================

class HSIPatchDataset(Dataset):
    def __init__(self, patches, scale=2):
        self.patches = patches
        self.scale = scale
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).permute(2, 0, 1).float()
        patch = patch.unsqueeze(0)  # Add channel dimension -> (1, B, H, W)
        hr = patch
        lr = torch.nn.functional.interpolate(
            hr, scale_factor=(1/self.scale, 1/self.scale), 
            mode='bilinear', align_corners=False
        )
        return lr, hr

def load_dataset(dataset_mat_path, patch_size=32):
    mat = sio.loadmat(dataset_mat_path)
    for k in ['indian_pines_corrected', 'indian_pines', 'paviaU']:
        if k in mat:
            data = mat[k].astype(np.float32)
            break
    else:
        for k, v in mat.items():
            if not k.startswith('__') and isinstance(v, np.ndarray):
                data = v.astype(np.float32)
                break
        else:
            raise KeyError("No valid hyperspectral dataset array found in the .mat file.")
            
    # Normalize to [0, 1]
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    h, w, b = data.shape
    
    # Create patches
    patches = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            
    return np.array(patches), data.shape

# ==========================================
# 4. EVALUATION UTILITIES
# ==========================================

def evaluate_metrics(hr_true, hr_pred):
    """
    hr_true, hr_pred: Numpy arrays of shape (H, W, Bands)
    """
    # PSNR
    psnrs = [psnr_func(hr_true[:,:,b], hr_pred[:,:,b], data_range=1.0) 
             for b in range(hr_true.shape[2])]
    avg_psnr = np.mean(psnrs)

    # SSIM
    ssims = [ssim_func(hr_true[:,:,b], hr_pred[:,:,b], data_range=1.0) 
             for b in range(hr_true.shape[2])]
    avg_ssim = np.mean(ssims)

    # SAM (degrees)
    vec_true = hr_true.reshape(-1, hr_true.shape[2])
    vec_pred = hr_pred.reshape(-1, hr_pred.shape[2])
    
    dot_product = np.sum(vec_true * vec_pred, axis=1)
    norms = np.linalg.norm(vec_true, axis=1) * np.linalg.norm(vec_pred, axis=1)
    
    angles = np.arccos(np.clip(dot_product / (norms + 1e-8), -1.0, 1.0))
    avg_sam = np.mean(angles) * (180.0 / np.pi)

    return avg_psnr, avg_ssim, avg_sam

# ==========================================
# 5. MAIN TRAINING PROCESS
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="SVNIT/ISRO HSI Super-Resolution CLI Trainer")
    parser.add_argument('--model', type=str, default='Lightweight3DCNN',
                        choices=['HSISuperRes3D', 'D3CNN', 'Lightweight3DCNN', 'Deep3DResNet', 'Dense3DNet', 'Hybrid3DNet'],
                        help="Model architecture to train")
    parser.add_argument('--dataset', type=str, default='dataset/Indian_pines_corrected.mat',
                        help="Path to HSI dataset .mat file")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--scale', type=int, default=2, help="Upsampling scale factor")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--save-dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("="*60)
    print("HSI SUPER-RESOLUTION TRAINING RUN")
    print("="*60)
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.dataset}")
    print(f"Scale:        {args.scale}x")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"LR:           {args.lr}")
    print(f"Patience:     {args.patience}")
    print("="*60)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:       {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:     {torch.cuda.get_device_name(0)}")

    # 2. Load Dataset
    try:
        patches, full_shape = load_dataset(args.dataset, patch_size=32)
        print(f"Successfully loaded dataset:")
        print(f"  Full HSI Shape:    {full_shape}")
        print(f"  Extracted Patches: {len(patches)} of size 32x32")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # Split dataset (80% train, 20% validation)
    indices = np.random.permutation(len(patches))
    split = int(0.8 * len(patches))
    train_patches = patches[indices[:split]]
    val_patches = patches[indices[split:]]
    print(f"  Training patches:  {len(train_patches)}")
    print(f"  Validation patches: {len(val_patches)}")

    train_dataset = HSIPatchDataset(train_patches, scale=args.scale)
    val_dataset = HSIPatchDataset(val_patches, scale=args.scale)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model Initialization
    bands_count = full_shape[2]
    if args.model == 'HSISuperRes3D':
        model = HSISuperRes3D(num_bands=bands_count)
    elif args.model == 'D3CNN':
        model = D3CNN(num_blocks=5)
    elif args.model == 'Lightweight3DCNN':
        model = Lightweight3DCNN(bands=bands_count)
    elif args.model == 'Deep3DResNet':
        model = Deep3DResNet(bands=bands_count, num_blocks=8)
    elif args.model == 'Dense3DNet':
        model = Dense3DNet(bands=bands_count)
    elif args.model == 'Hybrid3DNet':
        model = Hybrid3DNet(bands=bands_count)
    
    model = model.to(device)

    # 4. Optimizer, Loss & Setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HybridLoss(sam_weight=0.1)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': [], 'val_sam': []}

    # 5. Training Loop
    print("\nStarting training loop...")
    print("-" * 85)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'PSNR (dB)':<10} | {'SSIM':<10} | {'SAM (deg)':<10} | {'Time (s)':<8}")
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()
            output = model(lr_batch)
            loss = criterion(output, hr_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * lr_batch.size(0)

        train_loss = train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        # Accumulate metrics
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        epoch_sam = 0.0
        
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                
                output = model(lr_batch)
                loss = criterion(output, hr_batch)
                val_loss += loss.item() * lr_batch.size(0)
                
                # Convert back to numpy (H, W, Bands) for evaluation
                for b_idx in range(lr_batch.size(0)):
                    # Shape: (B, H, W)
                    hr_true_np = hr_batch[b_idx].squeeze(0).cpu().permute(1, 2, 0).numpy()
                    hr_pred_np = output[b_idx].squeeze(0).cpu().permute(1, 2, 0).numpy()
                    
                    psnr, ssim, sam = evaluate_metrics(hr_true_np, hr_pred_np)
                    epoch_psnr += psnr
                    epoch_ssim += ssim
                    epoch_sam += sam

        val_loss = val_loss / len(val_dataset)
        avg_psnr = epoch_psnr / len(val_dataset)
        avg_ssim = epoch_ssim / len(val_dataset)
        avg_sam = epoch_sam / len(val_dataset)

        elapsed = time.time() - start_time
        print(f"{epoch:<6d} | {train_loss:<10.6f} | {val_loss:<10.6f} | {avg_psnr:<10.2f} | {avg_ssim:<10.4f} | {avg_sam:<10.2f} | {elapsed:<8.1f}")

        # Save Metrics History
        history['epoch'].append(int(epoch))
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_psnr'].append(float(avg_psnr))
        history['val_ssim'].append(float(avg_ssim))
        history['val_sam'].append(float(avg_sam))

        # Early Stopping and Checkpoints
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save Best Model Checkpoint
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': {'psnr': float(avg_psnr), 'ssim': float(avg_ssim), 'sam': float(avg_sam)}
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[INFO] Early stopping triggered at epoch {epoch} (best validation loss: {best_loss:.6f})")
                break

    # Save final history to json
    history_path = os.path.join(args.save_dir, f"{args.model}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    total_time = time.time() - start_time
    print("-" * 85)
    print(f"Training Complete! Total Time: {total_time:.2f}s")
    print(f"Best Validation Loss:  {best_loss:.6f}")
    print(f"Saved Checkpoints to:  {args.save_dir}")
    print("-" * 85)

if __name__ == "__main__":
    main()
