import torch
import cv2
import os
import glob
import numpy as np

# 直接從你剛建立的純淨版資料夾匯入模型
from archs.basicvsrpp.basicvsr_pp import BasicVSRPlusPlus

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips # LPIPS 官方套件

from thop import profile
import time

print("🔄 正在初始化 BasicVSR++ (純淨 Bare-Metal 模式)...")

# 1. 初始化模型 (預設就是 3-Channel)
model = BasicVSRPlusPlus(
    mid_channels=64,
    num_blocks=7,
    is_low_res_input=True
)

# 2. 載入權重 (請確認檔名與你 checkpoints 資料夾內的一致)
ckpt_path = '/home/larry/ssd_data/sr_project/checkpoints/basicvsrpp/best_model_2.pth'
print(f"📦 正在讀取權重: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location='cuda')
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 過濾掉 'generator.' 前綴 (如果權重是從 mmagic 官方載下來的話通常會有)
clean_state_dict = {}
for k, v in state_dict.items():
    # 🚀 新增過濾機制：跳過 step_counter 這種無關網路架構的變數
    if k == 'step_counter' or k == 'meta':
        continue
        
    new_k = k.replace('generator.', '') if k.startswith('generator.') else k
    clean_state_dict[new_k] = v

model.load_state_dict(clean_state_dict, strict=True)
model = model.cuda().eval()

# 3. 準備輸入與輸出路徑
input_dir = '/home/larry/ssd_data/sr_project/datasets/test_frames'
out_dir = '/home/larry/ssd_data/sr_project/experiments/basicvsrpp/spaceship_test'
os.makedirs(out_dir, exist_ok=True)

# 4. 讀取太空船圖片
print(f"📂 正在讀取圖片從: {input_dir}")
img_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
if not img_paths:
    raise FileNotFoundError(f"找不到圖片，請檢查路徑: {input_dir}")

# 轉換成 [1, T, 3, H, W] 的 Tensor，並且歸一化到 0~1
imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in img_paths]
inputs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).unsqueeze(0).cuda()

print(f"🚀 執行推論... 輸入維度: {inputs.shape}")

# 1. 計算參數數量 (Parameters)
# 只計算需要更新的權重，單位換算成百萬 (M)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 模型參數數量 (Parameters): {total_params / 1e6:.2f} M")

# 2. 計算運算量 (FLOPs)
# thop 會跑一次 forward，我們用 10 幀的輸入去算
print("🧮 正在估算 FLOPs...")
macs, _ = profile(model, inputs=(inputs, ), verbose=False)
# MACs 轉換為 FLOPs 通常是 x2，並換算成 G (Giga) 單位
# 因為輸入是 10 幀，我們除以 10 算出「平均單幀」的 FLOPs 比較符合學術規範
flops_per_frame = (macs * 2) / 1e9 / inputs.size(1) 
print(f"⚡ 單幀運算量 (FLOPs/Frame): {flops_per_frame:.2f} G")

# 3. 計算推論速度 (Runtime / FPS)
# 在 GPU 測速必須使用 CUDA Event 才會精準
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    # 預熱 (Warm-up)：先空跑一次，喚醒 GPU，這樣測出來的時間才準
    _ = model(inputs)
    
    # 正式測速
    start_event.record()
    outputs = model(inputs)
    end_event.record()
    
    # 等待 GPU 把所有任務做完
    torch.cuda.synchronize()

runtime_ms = start_event.elapsed_time(end_event)
# 計算處理這 10 幀的平均單幀耗時與 FPS
ms_per_frame = runtime_ms / inputs.size(1)
fps = 1000.0 / ms_per_frame

print(f"⏱️ 執行速度:")
print(f"   - 總耗時 ({inputs.size(1)} 幀): {runtime_ms:.2f} ms")
print(f"   - 單幀耗時: {ms_per_frame:.2f} ms")
print(f"   - FPS: {fps:.2f}")
print(f"📐 輸出維度: {outputs.shape}")

# 5. 儲存高畫質圖片
print("💾 正在儲存放大的圖片...")
outputs_np = outputs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
for i, out_img in enumerate(outputs_np):
    # 轉回 BGR 格式並還原成 0-255 的數值
    out_bgr = cv2.cvtColor((out_img * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    save_path = os.path.join(out_dir, f"spaceship_out_{i:04d}.png")
    cv2.imwrite(save_path, out_bgr)

print(f"✅ 完成！圖片已成功儲存至 {out_dir}")


print("\n📊 開始計算評估指標...")

# 📍 這裡請換成你「真實高畫質原圖 (GT)」的路徑
gt_dir = '/home/larry/ssd_data/sr_project/datasets/test_gt' 
gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

if not gt_paths:
    print("⚠️ 找不到 GT 原圖，無法計算 PSNR/SSIM/LPIPS，僅計算 Pixel Std。")
    # Pixel Std 不需要原圖，直接算整張輸出的像素標準差
    # 這是衡量影像對比度/紋理豐富度的一個粗略指標
    pixel_std = outputs.std().item()
    print(f"📈 整體 Pixel Std: {pixel_std:.4f}")
else:
    # 讀取 GT 圖片並轉換為 Tensor [1, 10, 3, H, W]
    gt_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in gt_paths]
    gts = torch.from_numpy(np.stack(gt_imgs)).permute(0, 3, 1, 2).unsqueeze(0).cuda()

    # 初始化評估工具 (全部放進 GPU 加速)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    # LPIPS 預設使用 alexnet 提取特徵，這是學術界最通用的設定
    lpips_fn = lpips.LPIPS(net='alex').cuda() 

    total_psnr, total_ssim, total_lpips, total_std = 0, 0, 0, 0
    num_frames = outputs.size(1)

    # 逐幀計算 (因為 LPIPS 需要 2D 圖片輸入 [B, C, H, W])
    for i in range(num_frames):
        out_frame = outputs[:, i, :, :, :] # [1, 3, H, W]
        gt_frame = gts[:, i, :, :, :]       # [1, 3, H, W]

        # 1. PSNR & SSIM (數值越高越好)
        psnr_val = psnr_fn(out_frame, gt_frame).item()
        ssim_val = ssim_fn(out_frame, gt_frame).item()
        
        # 2. LPIPS (數值越低代表越接近人類視覺感知)
        # lpips 套件吃的是 [-1, 1] 的數值，所以要做簡單的正規化轉換
        out_frame_lpips = out_frame * 2.0 - 1.0
        gt_frame_lpips = gt_frame * 2.0 - 1.0
        lpips_val = lpips_fn(out_frame_lpips, gt_frame_lpips).item()
        
        # 3. Pixel Std (衡量對比度)
        std_val = out_frame.std().item()

        print(f"Frame {i:02d} | PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}, Std: {std_val:.4f}")

        total_psnr += psnr_val
        total_ssim += ssim_val
        total_lpips += lpips_val
        total_std += std_val

    print("-" * 40)
    print("🏆 序列平均成績:")
    print(f"Mean PSNR:  {total_psnr / num_frames:.2f} dB")
    print(f"Mean SSIM:  {total_ssim / num_frames:.4f}")
    print(f"Mean LPIPS: {total_lpips / num_frames:.4f}")
    print(f"Mean Std:   {total_std / num_frames:.4f}")