import torch
import torchvision
import os
from tqdm import tqdm
from models.FrMamba import FrMamba
import dataloader
import argparse


# ======================================================
# Test dataset paths
# ======================================================
DATASET_PATHS = [
    "./test/EUVP",
    "./test/LSUI",
    "./test/SEA-THRU",
    "./test/UIEB",
    "./test/Color-Check7"
]
# ======================================================


@torch.no_grad()
def test_single_dataset(model, dataset_path, output_root, batch_size=1, num_workers=0, device="cuda"):
    """Run inference on a single dataset and save results"""
    val_dataset = dataloader.val_loader(dataset_path)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dataset_name = os.path.basename(dataset_path.rstrip("/"))
    save_dir = os.path.join(output_root, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=== Testing dataset: {dataset_name} ===")

    for i, (img_in, names) in tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Inference [{dataset_name}]",
        dynamic_ncols=True
    ):
        img_in = img_in.to(device, non_blocking=True)
        clean_image = model(img_in)

        if isinstance(names, (list, tuple)):
            base_name = os.path.splitext(names[0])[0]
        else:
            base_name = os.path.splitext(names)[0]

        save_path = os.path.join(save_dir, f"{base_name}.png" if batch_size == 1 else f"{i + 1}.png")
        torchvision.utils.save_image(clean_image, save_path)

    print(f"✅ Done: results saved in {save_dir}")


@torch.no_grad()
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")

    # === Load model ===
    model = FrMamba().to(device)
    ckpt = torch.load(config.model_path, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"Loaded model weights from: {config.model_path}")

    os.makedirs(config.output_folder, exist_ok=True)

    # === Inference on multiple datasets ===
    for dataset_path in DATASET_PATHS:
        if os.path.exists(dataset_path):
            test_single_dataset(
                model=model,
                dataset_path=dataset_path,
                output_root=config.output_folder,
                batch_size=config.val_batch_size,
                num_workers=config.num_workers,
                device=device
            )
        else:
            print(f"Warning: dataset not found -> {dataset_path}")


# ======================================================
# Command line entry
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FrMamba Multi-Dataset Testing")
    parser.add_argument('--model_path', type=str, default="./snapshots/Epoch_200.pth")
    parser.add_argument('--output_folder', type=str, default="./results/")
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    config = parser.parse_args()

    main(config)
