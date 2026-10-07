import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from archs.edvr.edvr import EDVRNet

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))

class QRISPDatasetEDVR(Dataset):
    def __init__(self, qrisp_root, split='train', num_frames=5, lr_patch_size=64, scale=4):
        self.qrisp_root, self.num_frames, self.lr_patch_size, self.scale, self.split = qrisp_root, num_frames, lr_patch_size, scale, split
        self.hr_patch_size = lr_patch_size * scale

        with open(os.path.join(qrisp_root, f'{split}_list.txt'), 'r') as f:
            self.scenes = [line.strip() for line in f if line.strip()]

        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.exr')

        for scene in self.scenes:
            lr_native_dir = os.path.join(qrisp_root, scene, '270p', 'Native')
            hr_native_dir = os.path.join(qrisp_root, scene, '1080p', 'Native')
            if not os.path.exists(lr_native_dir) or not os.path.exists(hr_native_dir): continue

            for angle in sorted([d for d in os.listdir(lr_native_dir) if os.path.isdir(os.path.join(lr_native_dir, d))]):
                lr_angle_dir, hr_angle_dir = os.path.join(lr_native_dir, angle), os.path.join(hr_native_dir, angle)
                if not os.path.exists(hr_angle_dir): continue

                lr_f = sorted([os.path.join(lr_angle_dir, f) for f in os.listdir(lr_angle_dir) if f.lower().endswith(valid_exts)])
                hr_f = sorted([os.path.join(hr_angle_dir, f) for f in os.listdir(hr_angle_dir) if f.lower().endswith(valid_exts)])
                if len(lr_f) < num_frames or len(lr_f) != len(hr_f): continue
                self.samples.append({'lr_frames': lr_f, 'hr_frames': hr_f})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_paths, hr_paths = self.samples[idx]['lr_frames'], self.samples[idx]['hr_frames']
        start_idx = random.randint(0, len(lr_paths) - self.num_frames) if self.split == 'train' else (len(lr_paths) - self.num_frames) // 2

        lr_imgs = [cv2.imread(p) for p in lr_paths[start_idx : start_idx + self.num_frames]]
        hr_center_img = cv2.imread(hr_paths[start_idx + self.num_frames // 2])

        if self.split == 'train':
            lr_h, lr_w, _ = lr_imgs[0].shape
            lr_top, lr_left = random.randint(0, lr_h - self.lr_patch_size), random.randint(0, lr_w - self.lr_patch_size)
            hr_top, hr_left = lr_top * self.scale, lr_left * self.scale

            lr_imgs = [img[lr_top : lr_top + self.lr_patch_size, lr_left : lr_left + self.lr_patch_size, :] for img in lr_imgs]
            hr_center_img = hr_center_img[hr_top : hr_top + self.hr_patch_size, hr_left : hr_left + self.hr_patch_size, :]

            if random.random() < 0.5:
                lr_imgs, hr_center_img = [cv2.flip(img, 1) for img in lr_imgs], cv2.flip(hr_center_img, 1)
            if random.random() < 0.5:
                lr_imgs, hr_center_img = [cv2.flip(img, 0) for img in lr_imgs], cv2.flip(hr_center_img, 0)
            if random.random() < 0.5:
                lr_imgs, hr_center_img = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in lr_imgs], cv2.rotate(hr_center_img, cv2.ROTATE_90_CLOCKWISE)

        lr_tensors = [torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1) for img in lr_imgs]
        hr_center_tensor = torch.from_numpy(cv2.cvtColor(hr_center_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        return torch.stack(lr_tensors), hr_center_tensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EDVRNet(num_frames=5, with_tsa=True).to(device)
    
    qrisp_root = '/home/larry/ssd_data/sr_project/datasets/QRISP'
    save_dir = '/home/larry/ssd_data/sr_project/checkpoints/edvr'
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = DataLoader(QRISPDatasetEDVR(qrisp_root, 'train', 5, 64), batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(QRISPDatasetEDVR(qrisp_root, 'val', 5, 64), batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = CharbonnierLoss().to(device)
    
    best_val_loss = float('inf')
    step_counter, tsa_iter = 0, 2000 
    
    for epoch in range(300):
        model.train()
        epoch_loss = 0
        for batch_idx, (lr_seq, hr_center) in enumerate(train_loader):
            lr_seq, hr_center = lr_seq.to(device), hr_center.to(device)
            
            if step_counter == 0:
                for k, v in model.named_parameters():
                    if 'fusion' not in k: v.requires_grad = False
            elif step_counter == tsa_iter:
                for v in model.parameters(): v.requires_grad = True
            
            optimizer.zero_grad()
            outputs = model(lr_seq)
            loss = criterion(outputs, hr_center)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
            optimizer.step()
            
            epoch_loss += loss.item()
            step_counter += 1
            if batch_idx % 5 == 0: print(f"Epoch [{epoch+1}/300] Batch {batch_idx} | Loss: {loss.item():.6f}")
            
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_seq_v, hr_center_v in val_loader:
                val_loss += criterion(model(lr_seq_v.to(device)), hr_center_v.to(device)).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1} 驗證 Loss: {avg_val:.6f}")
        torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pth'))
        if avg_val < best_val_loss:
            best_val_loss, _ = avg_val, torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("🏆 已更新 best_model.pth")

if __name__ == '__main__':
    main()
