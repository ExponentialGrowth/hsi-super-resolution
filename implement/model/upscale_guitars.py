import os
import time
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def main():
    print("=" * 60)
    print("SVNIT/ISRO GUITAR IMAGE UPSCALER")
    print("=" * 60)
    
    # 1. Setup paths
    base_dir = r"e:\projects\svnit isro\implement\model"
    guitar_dir = os.path.join(base_dir, "guitar")
    output_dir = os.path.join(base_dir, "output of img incRes")
    weights_dir = os.path.join(base_dir, "weights")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Setup device
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    print(f"CUDA GPU Available: {is_cuda}")
    if is_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU. Note: Upscaling will be significantly slower on CPU.")
    
    # 3. Load model (Scale = 4)
    scale = 4
    model_path = os.path.join(weights_dir, f"RealESRGAN_x{scale}_fixed.pth")
    if not os.path.exists(model_path):
        # Fallback to non-fixed if fixed doesn't exist
        model_path = os.path.join(weights_dir, f"RealESRGAN_x{scale}.pth")
        
    print(f"Loading weights from: {model_path}")
    
    # Auto-fix weights dictionary structure if needed
    fixed_path = os.path.join(weights_dir, f"RealESRGAN_x{scale}_fixed.pth")
    if not os.path.exists(fixed_path) and os.path.exists(model_path):
        print("Adapting weights dictionary keys...")
        loadnet = torch.load(model_path, map_location='cpu')
        if 'params' not in loadnet and 'params_ema' not in loadnet:
            torch.save({'params': loadnet}, fixed_path)
            model_path = fixed_path
            print(f"Fixed weights saved to: {fixed_path}")
            
    # Initialize RRDBNet architecture
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    # Initialize the upsampler with tiling to prevent VRAM OOM
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,           # Fit in standard VRAM (4GB+)
        tile_pad=10,
        pre_pad=0,
        half=is_cuda,       # FP16 half-precision only on CUDA
        device=device
    )
    
    # 4. Find all images in guitar directory
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(guitar_dir) if f.lower().endswith(valid_exts)]
    images.sort()
    
    print(f"Found {len(images)} images to process in: {guitar_dir}")
    print("-" * 60)
    
    total_start = time.time()
    
    for idx, filename in enumerate(images, 1):
        img_path = os.path.join(guitar_dir, filename)
        # Output is PNG as configured in original notebook
        out_name = f"upscaled_{os.path.splitext(filename)[0]}.png"
        out_path = os.path.join(output_dir, out_name)
        
        if os.path.exists(out_path):
            print(f"[{idx}/{len(images)}] Skipping {filename} (Already upscaled)")
            continue
            
        print(f"[{idx}/{len(images)}] Processing {filename}...")
        start_time = time.time()
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # Upscale
            sr_img_np, _ = upsampler.enhance(img_np, outscale=scale)
            sr_img = Image.fromarray(sr_img_np)
            
            # Save output
            sr_img.save(out_path, format='PNG')
            elapsed = time.time() - start_time
            print(f"   [OK] Saved to {out_name} | Size: {sr_img.size} | Time: {elapsed:.2f}s")
            
            # Free memory
            del img, img_np, sr_img_np, sr_img
            if is_cuda:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   [ERROR] Error processing {filename}: {e}")
            
    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"Batch upscaling complete! Total time: {total_elapsed:.2f}s")
    print(f"Results saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
