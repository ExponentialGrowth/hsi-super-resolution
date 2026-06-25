import torch
import torch.nn as nn

class HSISuperRes3D(nn.Module):
    def __init__(self, in_channels=1, num_bands=103):
        super(HSISuperRes3D, self).__init__()
        
        # 3D Convolution: (Batch, Channels, Bands, Height, Width)
        # We use a small 3x3x3 kernel to capture local spectral-spatial features
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layer: Using PixelShuffle (scaled for 3D or simple interpolation)
        # Here we use Upsample for simplicity in a 3D context
        self.upsampler = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        
        self.reconstruction = nn.Conv3d(64, in_channels, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x):
        # x shape: (Batch, 1, Bands, H_low, W_low)
        residual = self.upsampler(x) 
        out = self.feature_extractor(x)
        out = self.upsampler(out)
        out = self.reconstruction(out)
        
        # Adding the residual helps the network learn only the "missing" details
        return out + residual

# Example usage:
# model = HSISuperRes3D(num_bands=103)
# input_cube = torch.randn(1, 1, 103, 32, 32) # Batch, Ch, Bands, H, W
# output_cube = model(input_cube) # Result: (1, 1, 103, 64, 64)