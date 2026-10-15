import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from torchmetrics.functional import structural_similarity_index_measure
from loss import *
from model import * 
from dataloader import * 

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Calculate PSNR
            psnr = calculate_psnr(output, high, gt_mean=True)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high, gt_mean=True)
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def main():
    import torch.optim as optim

    # Hyperparameters
    train_low = '/content/data/LOLdataset/our485/low'
    train_high = '/content/data/LOLdataset/our485/high'
    test_low = '/content/data/LOLdataset/eval15/low'
    test_high = '/content/data/LOLdataset/eval15/high'
    learning_rate = 4e-4
    num_epochs = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')


    model = DULL().to(device)


    # Initialize optimizer to only update fusion_module parameters
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    use_checkpoint = True

    epoch_c = 0 # Initialize epoch_c
    best_psnr = 0.0 # Initialize best_psnr

    if use_checkpoint:
        checkpoint_path = '/content/drive/MyDrive/Checkpoints/best_model_lolv1_2636.pth'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Check if the checkpoint is in the new dictionary format (with 'model_state_dict' key)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # epoch_c = checkpoint['epoch'] + 1 # Resume from the next epoch
                # if 'best_psnr' in checkpoint:
                #     best_psnr = checkpoint['best_psnr']
                # print(f"Loaded full checkpoint from epoch {checkpoint['epoch']}, with best PSNR: {best_psnr:.6f}.")
            else: # Assume it's the old format where only model.state_dict() was saved directly
                model.load_state_dict(checkpoint)
                print("Loaded old format model state dictionary directly. Optimizer, epoch, and best_psnr are NOT restored. Training will start from scratch with reinitialized optimizer.")
                # Since optimizer state is not loaded, ensure epoch_c and best_psnr are reset
                epoch_c = 0
                best_psnr = 0.0
                # Reinitialize optimizer and scheduler if their states weren't loaded
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
            epoch_c = 0
            best_psnr = 0.0
        except Exception as e: # Catch any other unexpected errors during loading
            print(f"An unexpected error occurred while loading checkpoint: {e}. Starting training from scratch.")
            epoch_c = 0
            best_psnr = 0.0

    print('Training started.')
    for epoch in range(epoch_c, num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(targets, outputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim= validate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs},Loss: {train_loss} PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss_at_save': train_loss, # Save the training loss at this epoch
                'best_psnr': best_psnr
            }, '/content/drive/MyDrive/Checkpoints/best_model_lolv1.pth')
            print(f'Saving model with PSNR: {best_psnr:.6f}')

if __name__ == '__main__':
    main()