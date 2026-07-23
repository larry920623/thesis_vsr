import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from archs.basicvsr.basicvsr import BasicVSRNet

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))

class QRISPDataset(Dataset):
    def __init__(self, qrisp_root, split='train', num_frames=15, lr_patch_size=64, scale=4):
        self.qrisp_root = qrisp_root
        self.num_frames = num_frames
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.scale = scale
        self.split = split

        list_path = os.path.join(qrisp_root, f'{split}_list.txt')
        with open(list_path, 'r') as f:
            self.scenes = [line.strip() for line in f if line.strip()]

        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.exr')

        for scene in self.scenes:
            scene_dir = os.path.join(qrisp_root, scene)
            lr_native_dir = os.path.join(scene_dir, '270p', 'Native')
            hr_native_dir = os.path.join(scene_dir, '1080p', 'Native')
            if not os.path.exists(lr_native_dir) or not os.path.exists(hr_native_dir): continue

            angles = [d for d in os.listdir(lr_native_dir) if os.path.isdir(os.path.join(lr_native_dir, d))]
            for angle in sorted(angles):
                lr_angle_dir = os.path.join(lr_native_dir, angle)
                hr_angle_dir = os.path.join(hr_native_dir, angle)
                if not os.path.exists(hr_angle_dir): continue

                lr_frames = sorted([os.path.join(lr_angle_dir, f) for f in os.listdir(lr_angle_dir) if f.lower().endswith(valid_exts)])
                hr_frames = sorted([os.path.join(hr_angle_dir, f) for f in os.listdir(hr_angle_dir) if f.lower().endswith(valid_exts)])
                if len(lr_frames) < num_frames or len(lr_frames) != len(hr_frames): continue

                self.samples.append({'lr_frames': lr_frames, 'hr_frames': hr_frames})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lr_paths, hr_paths = sample['lr_frames'], sample['hr_frames']
        total_frames = len(lr_paths)
        start_idx = random.randint(0, total_frames - self.num_frames) if self.split == 'train' else (total_frames - self.num_frames) // 2

        lr_imgs = [cv2.imread(p) for p in lr_paths[start_idx : start_idx + self.num_frames]]
        hr_imgs = [cv2.imread(p) for p in hr_paths[start_idx : start_idx + self.num_frames]]

        if self.split == 'train':
            lr_h, lr_w, _ = lr_imgs[0].shape
            lr_top = random.randint(0, lr_h - self.lr_patch_size)
            lr_left = random.randint(0, lr_w - self.lr_patch_size)
            hr_top, hr_left = lr_top * self.scale, lr_left * self.scale

            lr_imgs = [img[lr_top : lr_top + self.lr_patch_size, lr_left : lr_left + self.lr_patch_size, :] for img in lr_imgs]
            hr_imgs = [img[hr_top : hr_top + self.hr_patch_size, hr_left : hr_left + self.hr_patch_size, :] for img in hr_imgs]

            if random.random() < 0.5:
                lr_imgs, hr_imgs = [cv2.flip(img, 1) for img in lr_imgs], [cv2.flip(img, 1) for img in hr_imgs]
            if random.random() < 0.5:
                lr_imgs, hr_imgs = [cv2.flip(img, 0) for img in lr_imgs], [cv2.flip(img, 0) for img in hr_imgs]
            if random.random() < 0.5:
                lr_imgs, hr_imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in lr_imgs], [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in hr_imgs]

        lr_tensors = [torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1) for img in lr_imgs]
        hr_tensors = [torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1) for img in hr_imgs]
        return torch.stack(lr_tensors), torch.stack(hr_tensors)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicVSRNet().to(device)
    
    qrisp_root = '/home/larry/ssd_data/sr_project/datasets/QRISP'
    save_dir = '/home/larry/ssd_data/sr_project/checkpoints/basicvsr'
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = DataLoader(QRISPDataset(qrisp_root, 'train', 15, 64), batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(QRISPDataset(qrisp_root, 'val', 15, 64), batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    criterion = CharbonnierLoss().to(device)
    
    best_val_loss = float('inf')
    step_counter = 0
    fix_iter = 5000 
    
    for epoch in range(150):
        model.train()
        epoch_loss = 0
        for batch_idx, (lr_seq, hr_seq) in enumerate(train_loader):
            lr_seq, hr_seq = lr_seq.to(device), hr_seq.to(device)
            
            if step_counter < fix_iter:
                for k, v in model.named_parameters():
                    if 'spynet' in k: v.requires_grad_(False)
            elif step_counter == fix_iter:
                model.requires_grad_(True)
            
            optimizer.zero_grad()
            outputs = model(lr_seq)
            loss = criterion(outputs, hr_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
            optimizer.step()
            
            epoch_loss += loss.item()
            step_counter += 1
            if batch_idx % 5 == 0: print(f"Epoch [{epoch+1}/150] Batch {batch_idx} | Loss: {loss.item():.6f}")
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_seq_v, hr_seq_v in val_loader:
                outputs_v = model(lr_seq_v.to(device))
                val_loss += criterion(outputs_v, hr_seq_v.to(device)).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1} 驗證 Loss: {avg_val:.6f}")
        torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pth'))
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("🏆 已更新 best_model.pth")

if __name__ == '__main__':
    main()
