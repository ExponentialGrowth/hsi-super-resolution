import os
import argparse
import time
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Valid image extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

def load_upscaler(scale, device, weights_dir='weights'):
    scale = int(scale)
    # Define RRDBNet architecture (standard for Real-ESRGAN)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    original_path = os.path.join(weights_dir, f"RealESRGAN_x{scale}.pth")
    fixed_path = os.path.join(weights_dir, f"RealESRGAN_x{scale}_fixed.pth")
    
    # Check if weights exist
    if not os.path.exists(fixed_path):
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Required weights file not found: {original_path}")
        
        print(f"Adapting weight format for {original_path}...")
        loadnet = torch.load(original_path, map_location='cpu')
        # Standardize weight format for basicsr / realesrgan
        if 'params' not in loadnet and 'params_ema' not in loadnet:
            torch.save({'params': loadnet}, fixed_path)
            model_path = fixed_path
        else:
            model_path = original_path
    else:
        model_path = fixed_path
        
    print(f"Loading model weights from: {model_path}")
    
    # Initialize the upsampler with spatial tiling to prevent VRAM OOM
    is_cuda = (device.type == 'cuda')
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,           # Standard tile size to fit in average VRAM (4GB+)
        tile_pad=10,        # Padding overlap to avoid seams
        pre_pad=0,
        half=is_cuda,       # FP16 precision only supported on CUDA
        device=device
    )
    return upsampler

def process_image(img_path, output_path, upsampler, scale, img_format):
    start_time = time.time()
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    # Perform enhancement
    sr_img_np, _ = upsampler.enhance(img_np, outscale=scale)
    sr_img = Image.fromarray(sr_img_np)
    
    # Save image
    sr_img.save(output_path, format=img_format.upper())
    elapsed = time.time() - start_time
    
    # Clean memory
    del img, img_np, sr_img_np, sr_img
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="SVNIT/ISRO Standalone 2D Image Super-Resolution Utility")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Path to input image file or directory containing images")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="Path to output image file or directory to save results")
    parser.add_argument('--scale', '-s', type=int, default=4, choices=[2, 4, 8],
                        help="Upscaling scale factor (2, 4, or 8, default: 4)")
    parser.add_argument('--tile', type=int, default=400,
                        help="Tile size for spatial division (default: 400)")
    parser.add_argument('--format', type=str, default='png', choices=['png', 'jpg', 'jpeg', 'webp'],
                        help="Output image file format (default: png)")
    parser.add_argument('--weights-dir', type=str, default='weights',
                        help="Directory containing the model weights (default: weights)")
    args = parser.parse_args()

    # 1. Setup Device
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    
    print("=" * 60)
    print("IMAGE SUPER-RESOLUTION TOOL")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Scale Factor: {args.scale}x")
    print(f"Format:       {args.format}")
    print(f"Device:       {device.type.upper()}")
    if is_cuda:
        print(f"GPU Model:    {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 2. Check input and output types
    is_input_dir = os.path.isdir(args.input)
    
    # Validate paths
    if not os.path.exists(args.input):
        print(f"[ERROR] Input path does not exist: {args.input}")
        return

    # Initialize model
    try:
        upsampler = load_upscaler(args.scale, device, args.weights_dir)
        # Update tile size if configured differently
        upsampler.tile_size = args.tile
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    total_start = time.time()

    if is_input_dir:
        # Batch folder processing
        os.makedirs(args.output, exist_ok=True)
        files = [f for f in os.listdir(args.input) if f.lower().endswith(VALID_EXTENSIONS)]
        files.sort()
        
        print(f"Found {len(files)} images to process in directory.")
        print("-" * 60)
        
        success_count = 0
        for idx, filename in enumerate(files, 1):
            img_path = os.path.join(args.input, filename)
            out_name = f"upscaled_{os.path.splitext(filename)[0]}.{args.format}"
            out_path = os.path.join(args.output, out_name)
            
            print(f"[{idx}/{len(files)}] Processing {filename}...")
            try:
                elapsed = process_image(img_path, out_path, upsampler, args.scale, args.format)
                print(f"   [OK] Saved to {out_name} | Time: {elapsed:.2f}s")
                success_count += 1
            except Exception as e:
                print(f"   [ERROR] Failed to process {filename}: {e}")
                
            if is_cuda:
                torch.cuda.empty_cache()
                
        total_elapsed = time.time() - total_start
        print("-" * 60)
        print(f"Batch completed! Successfully processed {success_count}/{len(files)} images.")
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Results saved in: {args.output}")
        print("=" * 60)
        
    else:
        # Single file processing
        print("Processing single image file...")
        # Ensure parent directory of output file exists
        out_parent = os.path.dirname(args.output)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
            
        try:
            elapsed = process_image(args.input, args.output, upsampler, args.scale, args.format)
            print(f"   [OK] Super-resolved image saved to: {args.output}")
            print(f"   Time taken: {elapsed:.2f}s")
        except Exception as e:
            print(f"[ERROR] Failed to upscale image: {e}")
        print("=" * 60)

if __name__ == "__main__":
    main()
