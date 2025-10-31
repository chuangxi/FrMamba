import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# === Training dataset: paired GT and UW images ===
class UnderwaterTrainDataset(Dataset):
    def __init__(self, gt_dir, uw_dir, img_size=256):
        self.gt_dir = gt_dir
        self.uw_dir = uw_dir
        self.img_size = img_size

        exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
        gt_files, uw_files = [], []
        for e in exts:
            gt_files += glob.glob(os.path.join(gt_dir, e))
            uw_files += glob.glob(os.path.join(uw_dir, e))

        gt_files.sort()
        uw_files.sort()

        # Ensure one-to-one pairing
        self.length = min(len(gt_files), len(uw_files))
        self.gt_files = gt_files[:self.length]
        self.uw_files = uw_files[:self.length]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        gt_img = Image.open(self.gt_files[index]).convert('RGB')
        uw_img = Image.open(self.uw_files[index]).convert('RGB')
        gt_tensor = self.transform(gt_img)
        uw_tensor = self.transform(uw_img)
        return gt_tensor, uw_tensor  # input image, ground truth image


# === Validation dataset: only UW input (no GT) ===
class UnderwaterValDataset(Dataset):
    def __init__(self, uw_dir, img_size=256):
        self.uw_dir = uw_dir
        self.img_size = img_size

        exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
        uw_files = []
        for e in exts:
            uw_files += glob.glob(os.path.join(uw_dir, e))

        uw_files.sort()
        self.uw_files = uw_files

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.uw_files)

    def __getitem__(self, index):
        uw_img = Image.open(self.uw_files[index]).convert('RGB')
        uw_tensor = self.transform(uw_img)
        name = os.path.basename(self.uw_files[index])
        return uw_tensor, name  # return filename for saving results


# === Loader interface wrappers ===
def train_loader(gt_path, uw_path, img_size=256):
    return UnderwaterTrainDataset(gt_path, uw_path, img_size)


def val_loader(uw_path, img_size=256):
    return UnderwaterValDataset(uw_path, img_size)
