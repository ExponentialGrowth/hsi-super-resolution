import torch
import torch.nn as nn

class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, img1, img2):
        """
        img1, img2: (Batch, Bands, H, W)
        Returns the average spectral angle across all pixels.
        """
        # Flatten spatial dimensions to (Batch, Bands, Pixels)
        img1 = img1.view(img1.shape[0], img1.shape[1], -1)
        img2 = img2.view(img2.shape[0], img2.shape[1], -1)

        # Compute dot product and norms along the Band dimension
        dot_product = torch.sum(img1 * img2, dim=1)
        img1_norm = torch.sqrt(torch.sum(img1**2, dim=1))
        img2_norm = torch.sqrt(torch.sum(img2**2, dim=1))

        # SAM formula: arccos( (A . B) / (|A| * |B|) )
        # We clamp to [-1, 1] to prevent NaN during arccos due to float precision
        cos_theta = dot_product / (img1_norm * img2_norm + 1e-8)
        sam_angles = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

        return torch.mean(sam_angles)