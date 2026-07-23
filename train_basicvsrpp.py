import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

from archs.basicvsrpp.basicvsr_pp import BasicVSRPlusPlus

# ==========================================
# 1. 損失函數 (Charbonnier Loss)
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps))

# ==========================================
# 2. 專屬 QRISP 階層結構的 PyTorch Dataset
# ==========================================
class QRISPDataset(Dataset):
    def __init__(self, qrisp_root, split='train', num_frames=15, lr_patch_size=96, scale=4):
        self.qrisp_root = qrisp_root
        self.num_frames = num_frames
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.scale = scale
        self.split = split

        list_path = os.path.join(qrisp_root, f'{split}_list.txt')
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"❌ 找不到地圖檔: {list_path}")

        with open(list_path, 'r') as f:
            self.scenes = [line.strip() for line in f if line.strip()]

        self.samples = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.exr')

        # 深入解析結構: 場景 -> 解析度 -> Native -> 角度(0000, 0001)
        for scene in self.scenes:
            scene_dir = os.path.join(qrisp_root, scene)
            lr_native_dir = os.path.join(scene_dir, '270p', 'Native')
            hr_native_dir = os.path.join(scene_dir, '1080p', 'Native')

            if not os.path.exists(lr_native_dir) or not os.path.exists(hr_native_dir):
                print(f"⚠️ 跳過 {scene}: 找不到 270p/Native 或 1080p/Native")
                continue

            # 抓取該場景下所有的角度軌跡 (例如 '0000', '0001')
            angles = [d for d in os.listdir(lr_native_dir) if os.path.isdir(os.path.join(lr_native_dir, d))]

            for angle in sorted(angles):
                lr_angle_dir = os.path.join(lr_native_dir, angle)
                hr_angle_dir = os.path.join(hr_native_dir, angle)

                if not os.path.exists(hr_angle_dir):
                    print(f"⚠️ 跳過 {scene}/{angle}: 找不到對應的 HR 角度資料夾")
                    continue

                lr_frames = sorted([os.path.join(lr_angle_dir, f) for f in os.listdir(lr_angle_dir) if f.lower().endswith(valid_exts)])
                hr_frames = sorted([os.path.join(hr_angle_dir, f) for f in os.listdir(hr_angle_dir) if f.lower().endswith(valid_exts)])

                if len(lr_frames) < num_frames:
                    print(f"⚠️ 跳過 {scene}/{angle}: 影格數太少 (LR有 {len(lr_frames)} 張)")
                    continue
                    
                if len(lr_frames) != len(hr_frames):
                    print(f"⚠️ 跳過 {scene}/{angle}: LR 與 HR 影格數量不一致")
                    continue

                # 將每個角度都當作一個獨立的訓練樣本
                self.samples.append({
                    'scene_angle': f"{scene}/{angle}",
                    'lr_frames': lr_frames,
                    'hr_frames': hr_frames
                })

        print(f"📦 QRISP [{split}] 集初始化完成，共有 {len(self.samples)} 個有效影片片段 (Sequences)。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lr_paths = sample['lr_frames']
        hr_paths = sample['hr_frames']
        total_frames = len(lr_paths)

        if self.split == 'train':
            start_idx = random.randint(0, total_frames - self.num_frames)
        else:
            start_idx = (total_frames - self.num_frames) // 2

        lr_paths_seg = lr_paths[start_idx : start_idx + self.num_frames]
        hr_paths_seg = hr_paths[start_idx : start_idx + self.num_frames]

        lr_imgs = [cv2.imread(p) for p in lr_paths_seg]
        hr_imgs = [cv2.imread(p) for p in hr_paths_seg]

        if self.split == 'train':
            lr_h, lr_w, _ = lr_imgs[0].shape
            lr_top = random.randint(0, lr_h - self.lr_patch_size)
            lr_left = random.randint(0, lr_w - self.lr_patch_size)
            hr_top = lr_top * self.scale
            hr_left = lr_left * self.scale

            lr_imgs = [img[lr_top : lr_top + self.lr_patch_size, lr_left : lr_left + self.lr_patch_size, :] for img in lr_imgs]
            hr_imgs = [img[hr_top : hr_top + self.hr_patch_size, hr_left : hr_left + self.hr_patch_size, :] for img in hr_imgs]

            if random.random() < 0.5:
                lr_imgs = [cv2.flip(img, 1) for img in lr_imgs]
                hr_imgs = [cv2.flip(img, 1) for img in hr_imgs]
            if random.random() < 0.5:
                lr_imgs = [cv2.flip(img, 0) for img in lr_imgs]
                hr_imgs = [cv2.flip(img, 0) for img in hr_imgs]
            if random.random() < 0.5:
                lr_imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in lr_imgs]
                hr_imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in hr_imgs]

        lr_tensors = []
        hr_tensors = []
        for lr_img, hr_img in zip(lr_imgs, hr_imgs):
            lr_rgb = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hr_rgb = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            lr_tensors.append(torch.from_numpy(lr_rgb).permute(2, 0, 1))
            hr_tensors.append(torch.from_numpy(hr_rgb).permute(2, 0, 1))

        return torch.stack(lr_tensors), torch.stack(hr_tensors)

# ==========================================
# 3. 核心訓練引擎
# ==========================================
def main():
    print("🚀 初始化 BasicVSR++ 訓練程式...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BasicVSRPlusPlus(is_low_res_input=True).to(device)
    
    qrisp_root = '/home/larry/ssd_data/sr_project/datasets/QRISP'
    
    train_dataset = QRISPDataset(qrisp_root, split='train', num_frames=15, lr_patch_size=64)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = QRISPDataset(qrisp_root, split='val', num_frames=15, lr_patch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    criterion = CharbonnierLoss().to(device)
    
    num_epochs = 150
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (lr_seq, hr_seq) in enumerate(train_loader):
            lr_seq, hr_seq = lr_seq.to(device), hr_seq.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_seq)
            
            loss = criterion(outputs, hr_seq)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
        
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\n🏁 Epoch {epoch+1} 訓練結束 | 平均訓練 Loss: {avg_train_loss:.6f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr_seq_v, hr_seq_v in val_loader:
                lr_seq_v, hr_seq_v = lr_seq_v.to(device), hr_seq_v.to(device)
                outputs_v = model(lr_seq_v)
                val_loss += criterion(outputs_v, hr_seq_v).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch+1} 驗證成績 | 平均驗證 Loss: {avg_val_loss:.6f}")
        
        os.makedirs('checkpoints/qrisp_baseline', exist_ok=True)
        torch.save(model.state_dict(), '/home/larry/ssd_data/sr_project/checkpoints/basicvsrpp/latest_model_2.pth')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), '/home/larry/ssd_data/sr_project/checkpoints/basicvsrpp/best_model_2.pth')
            print("🏆 偵測到更低的驗證 Loss，已更新 best_model_2.pth ！")
        print("-" * 50)

if __name__ == '__main__':
    main()
