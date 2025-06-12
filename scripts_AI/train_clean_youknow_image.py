
import argparse, sys, random
from pathlib import Path
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
import cv2
import yaml

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts_AI.models import ResnetGenerator
from scripts_AI.model_util import NLayerDiscriminatorFromBVD, GANLoss, set_requires_grad
from scripts_AI import image_processing_util as ipu

class PairedCleanDataset(Dataset):
    def __init__(self, data_path, out_size=256):
        self.data_path = Path(data_path)
        self.mosaiced_dir = self.data_path / "mosaiced"
        self.origin_dir = self.data_path / "origin_image"

        if not self.mosaiced_dir.exists() or not self.origin_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục 'mosaiced' hoặc 'origin_image' trong '{data_path}'. Chạy 'train.py' để tự động tạo.")

        self.image_list = sorted([p.name for p in self.mosaiced_dir.glob('*.*')])
        self.out_size = out_size

    def __len__(self): return len(self.image_list)
    def __getitem__(self, idx):
        name = self.image_list[idx]
        img_mosaiced = ipu.imread(str(self.mosaiced_dir / name), rgb=True)
        img_origin = ipu.imread(str(self.origin_dir / name), rgb=True)

        if img_mosaiced.shape[0] != self.out_size: img_mosaiced = cv2.resize(img_mosaiced, (self.out_size, self.out_size))
        if img_origin.shape[0] != self.out_size: img_origin = cv2.resize(img_origin, (self.out_size, self.out_size))

        img_mosaiced = (img_mosaiced.astype(np.float32) / 127.5) - 1.0
        img_origin = (img_origin.astype(np.float32) / 127.5) - 1.0

        return torch.from_numpy(img_mosaiced.transpose(2,0,1)), torch.from_numpy(img_origin.transpose(2,0,1))

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện model làm sạch ảnh (pix2pix-style)")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--folder_path', required=True, type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--lambda_L1', type=float, default=100.0)
    args, _ = parser.parse_known_args()

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")
    dir_checkpoint = Path(args.folder_path)

    dataset = PairedCleanDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    netG = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9).to(device)
    netD = NLayerDiscriminatorFromBVD(input_nc=6).to(device)

    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion_GAN = GANLoss(gan_mode='lsgan').to(device)
    criterion_L1 = nn.L1Loss()

    print("--- Bắt đầu huấn luyện 'clean_youknow_image' ---")
    for epoch in range(args.epochs):
        for real_A, real_B in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            real_A, real_B = real_A.to(device), real_B.to(device)
            fake_B = netG(real_A)
            set_requires_grad(netD, True); optimizer_D.zero_grad()
            pred_real = netD(torch.cat((real_A, real_B), 1)); loss_D_real = criterion_GAN(pred_real, True)
            pred_fake = netD(torch.cat((real_A, fake_B.detach()), 1)); loss_D_fake = criterion_GAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward(); optimizer_D.step()
            set_requires_grad(netD, False); optimizer_G.zero_grad()
            pred_fake_G = netD(torch.cat((real_A, fake_B), 1)); loss_G_GAN = criterion_GAN(pred_fake_G, True)
            loss_G_L1 = criterion_L1(fake_B, real_B) * args.lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward(); optimizer_G.step()

        print(f"Epoch {epoch+1} done. Loss G: {loss_G.item():.4f}, Loss D: {loss_D.item():.4f}")

    print("Huấn luyện hoàn tất.")
    model_save_path = dir_checkpoint / "clean_youknow_image.pth"
    torch.save(netG.state_dict(), model_save_path)

    arch_info = {'architecture': 'ResnetGenerator', 'params': {'n_blocks': 9}}
    yaml_save_path = dir_checkpoint / "clean_youknow_image_structure.yaml"
    with open(yaml_save_path, 'w') as f: yaml.dump(arch_info, f)
    print(f"Model đã được lưu tại: {model_save_path}")

if __name__ == '__main__':
    main()
