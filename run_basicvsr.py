import os
import torch
import cv2
import glob
import numpy as np

from archs.basicvsr.basicvsr import BasicVSRNet
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from thop import profile
import time

def main():
    print("🔄 正在初始化 BasicVSR (純淨 Bare-Metal 模式)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化模型
    model = BasicVSRNet(mid_channels=64, num_blocks=30).to(device)

    # 2. 載入權重
    ckpt_path = '/home/larry/ssd_data/sr_project/checkpoints/basicvsr/best_model.pth'
    print(f"📦 正在讀取權重: {ckpt_path}")
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # 過濾掉 'generator.' 前綴與無關網路架構的變數
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k == 'step_counter' or k == 'meta':
                continue
            new_k = k.replace('generator.', '') if k.startswith('generator.') else k
            clean_state_dict[new_k] = v
            
        model.load_state_dict(clean_state_dict, strict=True)
        print("✅ 成功載入 best_model.pth")
    else:
        print("⚠️ 找不到 best_model.pth，請先執行訓練。")
        return
        
    model.eval()

    # 3. 準備輸入與輸出路徑
    input_dir = '/home/larry/ssd_data/sr_project/datasets/test_frames'
    gt_dir = '/home/larry/ssd_data/sr_project/datasets/test_gt'
    out_dir = '/home/larry/ssd_data/sr_project/experiments/basicvsr/spaceship_test'
    os.makedirs(out_dir, exist_ok=True)

    # 4. 讀取太空船圖片
    print(f"📂 正在讀取圖片從: {input_dir}")
    img_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    if not img_paths:
        raise FileNotFoundError(f"找不到圖片，請檢查路徑: {input_dir}")

    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in img_paths]
    inputs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    print(f"🚀 執行推論... 輸入維度: {inputs.shape}")

    # --- 硬體效能與運算量評估 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 模型參數數量 (Parameters): {total_params / 1e6:.2f} M")

    print("🧮 正在估算 FLOPs...")
    macs, _ = profile(model, inputs=(inputs, ), verbose=False)
    flops_per_frame = (macs * 2) / 1e9 / inputs.size(1)
    print(f"⚡ 單幀運算量 (FLOPs/Frame): {flops_per_frame:.2f} G")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        _ = model(inputs) # 預熱
        start_event.record()
        outputs = model(inputs)
        end_event.record()
        torch.cuda.synchronize()

    runtime_ms = start_event.elapsed_time(end_event)
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
        out_bgr = cv2.cvtColor((out_img * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        save_path = os.path.join(out_dir, f"{i:04d}.png") # 儲存格式對齊 0000.png 
        cv2.imwrite(save_path, out_bgr)
    print(f"✅ 完成！圖片已成功儲存至 {out_dir}")

    # --- 影像品質評估指標計算 ---
    print("\n📊 開始計算評估指標...")
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))

    if not gt_paths:
        print("⚠️ 找不到 GT 原圖，無法計算 PSNR/SSIM/LPIPS，僅計算 Pixel Std。")
        pixel_std = outputs.std().item()
        print(f"📈 整體 Pixel Std: {pixel_std:.4f}")
    else:
        gt_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in gt_paths]
        gts = torch.from_numpy(np.stack(gt_imgs)).permute(0, 3, 1, 2).unsqueeze(0).to(device)

        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_fn = lpips.LPIPS(net='alex').to(device)

        total_psnr, total_ssim, total_lpips, total_std = 0, 0, 0, 0
        num_frames = outputs.size(1)

        for i in range(num_frames):
            out_frame = outputs[:, i, :, :, :]
            gt_frame = gts[:, i, :, :, :]

            psnr_val = psnr_fn(out_frame, gt_frame).item()
            ssim_val = ssim_fn(out_frame, gt_frame).item()
            
            out_frame_lpips = out_frame * 2.0 - 1.0
            gt_frame_lpips = gt_frame * 2.0 - 1.0
            lpips_val = lpips_fn(out_frame_lpips, gt_frame_lpips).item()
            
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

if __name__ == '__main__':
    main()
