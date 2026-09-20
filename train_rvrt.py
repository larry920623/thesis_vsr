import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# 假設你已經將 network_rvrt.py 放在 archs/rvrt/ 底下，並編譯好 deform_attn
from archs.rvrt.network_rvrt import RVRT

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))

class QRISPDatasetRVRT(Dataset):
    def __init__(self, qrisp_root, split='train', num_frames=8, lr_patch_size=64, scale=4):
        self.qrisp_root, self.num_frames, self.lr_patch_size, self.scale, self.split = qrisp_root, num_frames, lr_patch_size, scale, split
        self.hr_patch_size = lr_patch_size * scale
        with open(os.path.join(qrisp_root, f'{split}_list.txt'), 'r') as f:
            self.scenes = [line.strip() for line in f if line.strip()]
        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.exr')
        for scene in self.scenes:
            lr_native_dir, hr_native_dir = os.path.join(qrisp_root, scene, '270p', 'Native'), os.path.join(qrisp_root, scene, '1080p', 'Native')
            if not os.path.exists(lr_native_dir) or not os.path.exists(hr_native_dir): continue
            for angle in sorted([d for d in os.listdir(lr_native_dir) if os.path.isdir(os.path.join(lr_native_dir, d))]):
                lr_dir, hr_dir = os.path.join(lr_native_dir, angle), os.path.join(hr_native_dir, angle)
                lr_f = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.lower().endswith(valid_exts)])
                hr_f = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.lower().endswith(valid_exts)])
                # 確保有足夠的幀數
                if len(lr_f) >= num_frames and len(lr_f) == len(hr_f): 
                    self.samples.append({'lr': lr_f, 'hr': hr_f})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        lr_paths, hr_paths = self.samples[idx]['lr'], self.samples[idx]['hr']
        start_idx = random.randint(0, len(lr_paths) - self.num_frames) if self.split == 'train' else (len(lr_paths) - self.num_frames) // 2
        lr_imgs = [cv2.imread(p) for p in lr_paths[start_idx : start_idx + self.num_frames]]
        hr_imgs = [cv2.imread(p) for p in hr_paths[start_idx : start_idx + self.num_frames]]

        if self.split == 'train':
            h, w = lr_imgs[0].shape[:2]
            top, left = random.randint(0, h - self.lr_patch_size), random.randint(0, w - self.lr_patch_size)
            lr_imgs = [img[top : top + self.lr_patch_size, left : left + self.lr_patch_size, :] for img in lr_imgs]
            hr_imgs = [img[top * self.scale : (top + self.lr_patch_size) * self.scale, left * self.scale : (left + self.lr_patch_size) * self.scale, :] for img in hr_imgs]
            
            if random.random() < 0.5: lr_imgs, hr_imgs = [cv2.flip(img, 1) for img in lr_imgs], [cv2.flip(img, 1) for img in hr_imgs]
            if random.random() < 0.5: lr_imgs, hr_imgs = [cv2.flip(img, 0) for img in lr_imgs], [cv2.flip(img, 0) for img in hr_imgs]
        else:
            # Validation Center Crop: 確保是 window_size (8) 的倍數，使用 128x128
            patch = 128
            h, w = lr_imgs[0].shape[:2]
            top, left = (h - patch) // 2, (w - patch) // 2
            lr_imgs = [img[top : top + patch, left : left + patch, :] for img in lr_imgs]
            hr_imgs = [img[top * self.scale : (top + patch) * self.scale, left * self.scale : (left + patch) * self.scale, :] for img in hr_imgs]

        return torch.stack([torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1) for img in lr_imgs]), \
               torch.stack([torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1) for img in hr_imgs])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 建立 RVRT 模型 (預設 clip_size=2, window_size=[2,8,8])
    # 首次執行會自動下載 spynet 權重
    model = RVRT(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8]).to(device)
    
    qrisp_root = '/home/larry/ssd_data/sr_project/datasets/QRISP'
    save_dir = '/home/larry/ssd_data/sr_project/checkpoints/rvrt'
    
    # 注意：RVRT 記憶體需求極高，batch_size 建議先設為 1，視 VRAM 情況再調高
    train_loader = DataLoader(QRISPDatasetRVRT(qrisp_root, 'train', num_frames=8, lr_patch_size=64), batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(QRISPDatasetRVRT(qrisp_root, 'val', num_frames=8, lr_patch_size=64), batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = CharbonnierLoss().to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(300):
        model.train()
        epoch_loss = 0
        for batch_idx, (lr_seq, hr_seq) in enumerate(train_loader):
            lr_seq, hr_seq = lr_seq.to(device), hr_seq.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_seq)
            loss = criterion(outputs, hr_seq)
            loss.backward()
            
            # 強制梯度裁切，防止 Deformable Attention 與 Transformer 崩潰
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50) 
            optimizer.step()
            
            epoch_loss += loss.item()
            if batch_idx % 5 == 0: print(f"Epoch [{epoch+1}/300] Batch {batch_idx} | Loss: {loss.item():.6f}")
            
        scheduler.step()
        torch.cuda.empty_cache()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_seq_v, hr_seq_v in val_loader:
                val_loss += criterion(model(lr_seq_v.to(device)), hr_seq_v.to(device)).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1} 驗證 Loss: {avg_val:.6f}")
        torch.save(model.state_dict(), os.path.join(save_dir, 'latest_model.pth'))
        if avg_val < best_val_loss:
            best_val_loss, _ = avg_val, torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("🏆 已更新 best_model.pth")

if __name__ == '__main__':
    main()
