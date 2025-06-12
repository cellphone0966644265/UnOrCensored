
import argparse, sys, random, time, yaml
from pathlib import Path
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts_AI.models import BiSeNet
from scripts_AI import image_processing_util as ipu

class SegmentationDataset(Dataset):
    def __init__(self, data_path, fine_size=360):
        self.fine_size = fine_size
        origin_dir = Path(data_path) / "origin_image"
        mask_dir = Path(data_path) / "mask"
        self.image_paths = sorted(list(origin_dir.glob('*.*')))
        self.mask_paths = sorted([mask_dir / p.name for p in self.image_paths])

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = ipu.imread(str(self.image_paths[idx]), rgb=True)
        mask = ipu.imread(str(self.mask_paths[idx]), mod='gray')
        h, w, _ = img.shape
        load_size = max(h,w, self.fine_size)
        img = cv2.resize(img, (load_size, load_size))
        mask = cv2.resize(mask, (load_size, load_size))
        top = random.randint(0, load_size - self.fine_size)
        left = random.randint(0, load_size - self.fine_size)
        img = img[top:top+self.fine_size, left:left+self.fine_size]
        mask = mask[top:top+self.fine_size, left:left+self.fine_size]
        img_tensor = (img.transpose((2, 0, 1)) / 255.0)
        mask_tensor = (mask.reshape(1, self.fine_size, self.fine_size) / 255.0)
        return torch.from_numpy(img_tensor).float(), torch.from_numpy(mask_tensor).float()

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện model BiSeNet để xác định vị trí")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--folder_path', required=True, type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    args, _ = parser.parse_known_args()

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")
    dir_checkpoint = Path(args.folder_path)
    dir_checkpoint.mkdir(exist_ok=True)

    dataset = SegmentationDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    net = BiSeNet(num_classes=1, context_path='resnet18', train_flag=True).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    print("--- Bắt đầu huấn luyện 'mosaic_position' ---")
    for epoch in range(args.epochs):
        net.train()
        total_loss = 0
        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            pred, sup1, sup2 = net(imgs)
            loss = criterion(pred, masks) + criterion(sup1, masks) + criterion(sup2, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

    model_save_path = dir_checkpoint / "mosaic_position.pth"
    torch.save(net.state_dict(), model_save_path)
    arch_info = {'architecture': 'BiSeNet', 'params': {'num_classes': 1, 'context_path': 'resnet18'}}
    yaml_save_path = dir_checkpoint / "mosaic_position_structure.yaml"
    with open(yaml_save_path, 'w') as f: yaml.dump(arch_info, f)
    print(f"Đã lưu model tại: {model_save_path}")

if __name__ == "__main__":
    main()
