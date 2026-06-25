from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import os
from super_resolve import visualize_sr

img = visualize_sr("./Dumas_Family/HR/Dumas_Family_HR.png", halve=True)
img.save("sr_visualization.png")
#python save_sr.py "./Dumas_Family/HR/Dumas_Family_HR.png" "./Dumas_Family/SR/Dumas_Family_SR.png" halve=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data_folder = "./"
test_data_names = ["Set5", "Set14", "BSDS100","Dumas_Family"]

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srresnet_checkpoint = "./checkpoint_srresnet.pth.tar"

# Check if checkpoint files exist
if not os.path.exists(srgan_checkpoint):
    print(f"ERROR: Checkpoint file '{srgan_checkpoint}' not found!")
    print("\nTo use this evaluation script, you need to:")
    print("1. Train the models using train_srresnet.py and train_srgan.py, OR")
    print("2. Download pre-trained checkpoints from:")
    print("   https://drive.google.com/drive/folders/12OG-KawSFFs6Pah89V4a_Td-VcwMBE5i?usp=sharing")
    print("\nFor training, you also need to:")
    print("- Download MSCOCO '14 training and validation images")
    print("- Download Set5, Set14, and BSD100 test datasets")
    print("- Run: python create_data_lists.py")
    print("- Run: python train_srresnet.py")
    print("- Run: python train_srgan.py")
    exit(1)

# Load model, either the SRResNet or the SRGAN
# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()
model = srgan_generator

# Evaluate
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader
    test_dataset = SRDataset(data_folder,
                             split='test',
                             crop_size=0,
                             scaling_factor=4,
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                              pin_memory=True)

    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # Forward prop.
            sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                           data_range=255.)
            ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                         data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))

    # Print average PSNR and SSIM
    print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

print("\n")
