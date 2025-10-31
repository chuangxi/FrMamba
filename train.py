import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from models.FrMamba import FrMamba
import numpy as np
from torch.nn import init
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
# import lpips

def load_pretrained_model(pretrained_path):
    model = FrMamba()
    pretrained_dict = torch.load(pretrained_path)
    if 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model']
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, "./"))

    # === Model ===
    model = FrMamba().to(device)

    # === Datasets ===
    train_dataset = dataloader.train_loader(
        config.train_gt_images_path,
        config.train_underwater_images_path
    )
    val_dataset = dataloader.val_loader(config.val_underwater_images_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # === Loss & Optimizer ===
    l2 = nn.MSELoss().to(device)
    l1 = nn.L1Loss().to(device)
    # loss_lpips = lpips.LPIPS(net='vgg').to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # === Warmup + CosineAnnealing Scheduler ===
    warmup_ratio = 0.05  # 前5% epoch用于warmup
    # warmup_epochs = max(1, int(config.num_epochs * warmup_ratio))
    warmup_epochs = max(1, 25)
    cosine_epochs = config.num_epochs - warmup_epochs

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,  # 初始学习率 = 0.1 * lr
        total_iters=warmup_epochs
    )

    eta_min = config.lr * 0.01  # 最小学习率设置为原lr的1%
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    print(f"[LR Scheduler] Warmup {warmup_epochs} epochs "
          f"({warmup_ratio*100:.0f}%), then CosineAnnealing for {cosine_epochs} epochs.")

    # === Metrics ===
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # === Training Loop ===
    model.train()
    for epoch in range(config.num_epochs):
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            dynamic_ncols=True,
            leave=False
        )

        for iteration, (img_orig, img_haze) in pbar:
            img_orig = img_orig.to(device, non_blocking=True)
            img_haze = img_haze.to(device, non_blocking=True)

            # === Forward ===
            clean_image = model(img_haze)

            # === Loss ===
            loss = (
                l1(clean_image, img_orig) 
                + l2(clean_image, img_orig) 
                + (1 - ssim(clean_image, img_orig)) 
                # + loss_lpips(clean_image, img_orig).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # === Progress ===
            pbar.set_postfix(iter=iteration + 1, loss=f"{loss.item():.4f}")

            if ((iteration + 1) % config.snapshot_iter) == 0:
                writer.add_scalar('Loss/train_iter', loss.item(), epoch * len(train_loader) + iteration)

        # === Update Scheduler ===
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        print(f"Epoch {epoch+1:03d}: lr = {current_lr:.6e}")

        # === Save Model ===
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join(config.snapshots_folder, f"Epoch_{epoch + 1}.pth"))
        torch.save(model.state_dict(),
                   os.path.join(config.snapshots_folder, "FrMamba.pth"))

        # === Validation ===
        model.eval()
        with torch.no_grad():
            vbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Val   {epoch+1}/{config.num_epochs}",
                dynamic_ncols=True,
                leave=False
            )
            for iter_val, (img_haze, names) in vbar:
                img_haze = img_haze.to(device, non_blocking=True)
                clean_image = model(img_haze)

                if isinstance(names, (list, tuple)):
                    base_name = os.path.splitext(names[0])[0]
                else:
                    base_name = os.path.splitext(names)[0]

                save_path = os.path.join(
                    config.sample_output_folder,
                    f"{base_name}.png" if config.val_batch_size == 1 else f"{iter_val + 1}.png"
                )

                torchvision.utils.save_image(
                    torch.cat((img_haze, clean_image), dim=0),
                    save_path
                )
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # === Input Parameters ===
    parser.add_argument('--train_gt_images_path', type=str, default="/root/4000/gt")
    parser.add_argument('--train_underwater_images_path', type=str, default="/root/4000/raw")
    parser.add_argument('--val_underwater_images_path', type=str, default="/root/adata/val/0")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip_norm', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--tensorboard_dir', type=str, default="/root/tf-logs/")
    config = parser.parse_args()

    # === Folder Setup ===
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.sample_output_folder, exist_ok=True)

    train(config)
