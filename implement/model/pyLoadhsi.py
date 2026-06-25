import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

file_path = "/dataset/Indian_pines_corrected.mat"
def prepare_hsi_data(file_path, patch_size=32):
    # 1. Load the .mat file
    # For Indian Pines, the key is usually 'indian_pines_corrected'
    data = sio.loadmat(file_path)['indian_pines_corrected']
    
    # 2. Pre-processing: Normalization
    # Reshape to 2D to normalize, then back to 3D
    h, w, b = data.shape
    data_reshaped = data.reshape(-1, b)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_reshaped)
    data = data_normalized.reshape(h, w, b)

    # 3. Create Patches (Sliding Window)
    patches = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            
    return np.array(patches)

# Usage
# patches = prepare_hsi_data('Indian_pines_corrected.mat')
# print(f"Generated {len(patches)} patches of shape {patches[0].shape}")