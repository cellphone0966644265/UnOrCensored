
import argparse, sys, time, random, yaml
from pathlib import Path
from collections import OrderedDict

import torch, torch.nn as nn, numpy as np, cv2
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts_AI.models import BVDNet
from scripts_AI.model_util import VGGLoss, HingeLossD, HingeLossG, MultiscaleDiscriminator, set_requires_grad
from scripts_AI import image_processing_util as ipu

class VideoCleanDataset(Dataset):
    def __init__(self, data_path, clip_len=100, finesize=256):
        self.clip_dirs = sorted([p for p in Path(data_path).iterdir() if p.is_dir()])
        self.clip_len = clip_len
        self.finesize = finesize
    def __len__(self): return len(self.clip_dirs)
    def __getitem__(self, idx):
        clip_path = self.clip_dirs[idx]
        origin_dir = clip_path / 'origin_image'; mask_dir = clip_path / 'mask'
        frame_names = sorted([p.name for p in origin_dir.glob('*.*')])
        if not frame_names or len(frame_names) < 10: return None, None
        if len(frame_names) < self.clip_len: frame_names = (frame_names * (self.clip_len // len(frame_names) + 1))
        start_idx = random.randint(0, len(frame_names) - self.clip_len)
        clip_frame_names = frame_names[start_idx : start_idx + self.clip_len]
        ori_frames = [ipu.imread(str(origin_dir / name), loadsize=self.finesize, rgb=True) for name in clip_frame_names]
        mask_frames = [ipu.imread(str(mask_dir / name), mod='gray', loadsize=self.finesize) for name in clip_frame_names]
        return np.array(ori_frames), np.array(mask_frames)

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện model làm sạch video (BVDNet)")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--folder_path', required=True, type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    args, _ = parser.parse_known_args()

    N, S, T = 2, 3, 5
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")
    dir_checkpoint = Path(args.folder_path)

    dataset = VideoCleanDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn_skip_none)

    netG = BVDNet(N=N, n_blocks=4).to(device)
    netD = MultiscaleDiscriminator(input_nc=6, n_layers_D=2, num_D=3).to(device)
    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion_L2 = nn.MSELoss()
    criterion_VGG = VGGLoss(gpu_id=args.gpu_id)
    criterion_GAN_G = HingeLossG()
    criterion_GAN_D = HingeLossD()

    print("--- Bắt đầu huấn luyện 'clean_youknow_video' ---")
    for epoch in range(args.epochs):
        for ori_clip_np, mask_clip_np in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            if ori_clip_np is None: continue
            ori_clip = ori_clip_np.float().to(device) / 127.5 - 1.0; mask_clip = mask_clip_np.float().to(device) / 255.0
            ori_clip = ori_clip.permute(0, 4, 1, 2, 3); mask_clip = mask_clip.permute(0, 3, 1, 2).unsqueeze(2)
            t_center = random.randint(N * S, ori_clip.shape[2] - N * S - 1)
            sequence_indices = [t_center + (i - N) * S for i in range(T)]
            ori_sequence = ori_clip[:, :, sequence_indices, :, :]; mask_sequence = mask_clip[:, sequence_indices, :, :, :]
            mosaic_sequence = ori_sequence * (1 - mask_sequence) + torch.randn_like(ori_sequence) * mask_sequence
            center_frame_ori = ori_sequence[:, :, N, :, :]; center_frame_mosaic = mosaic_sequence[:, :, N, :, :]
            previous_frame = ori_clip[:, :, t_center - 1, :, :]

            set_requires_grad(netD, False); optimizer_G.zero_grad()
            fake_frame = netG(mosaic_sequence, previous_frame)
            loss_G = criterion_GAN_G(netD(torch.cat((center_frame_mosaic, fake_frame), dim=1))) + \
                     criterion_L2(fake_frame, center_frame_ori) * 100.0 + \
                     criterion_VGG(fake_frame, center_frame_ori) * 10.0
            loss_G.backward(); optimizer_G.step()

            set_requires_grad(netD, True); optimizer_D.zero_grad()
            loss_D = criterion_GAN_D(netD(torch.cat((center_frame_mosaic, fake_frame.detach()), dim=1)),
                                     netD(torch.cat((center_frame_mosaic, center_frame_ori), dim=1)))
            loss_D.backward(); optimizer_D.step()

        print(f"Epoch {epoch+1} done. Loss G: {loss_G.item():.4f}, Loss D: {loss_D.item():.4f}")

    print("Huấn luyện hoàn tất.")
    model_save_path = dir_checkpoint / "clean_youknow_video.pth"
    torch.save(netG.state_dict(), model_save_path)

    arch_info = {'architecture': 'BVDNet', 'params': {'N': N, 'n_blocks': 4}}
    yaml_save_path = dir_checkpoint / "clean_youknow_video_structure.yaml"
    with open(yaml_save_path, 'w') as f: yaml.dump(arch_info, f)
    print(f"Model đã được lưu tại: {model_save_path}")

if __name__ == '__main__':
    main()
