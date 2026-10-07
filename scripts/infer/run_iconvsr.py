import os
import torch
import cv2
import glob
import numpy as np
from archs.iconvsr.iconvsr import IconVSRNet
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from thop import profile

def main():
    print("🔄 正在初始化 IconVSR (純淨 Bare-Metal 模式)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IconVSRNet().to(device)
    
    ckpt_path = '/home/larry/ssd_data/sr_project/checkpoints/iconvsr/best_model.pth'
    if os.path.exists(ckpt_path):
        clean_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(ckpt_path, map_location=device).get('state_dict', torch.load(ckpt_path)).items() if k not in ['step_counter', 'meta']}
        model.load_state_dict(clean_state_dict, strict=True)
        print("✅ 成功載入 best_model.pth")
    else:
        print("⚠️ 找不到 best_model.pth，請先執行訓練。")
        return
        
    model.eval()

    input_dir, gt_dir = '/home/larry/ssd_data/sr_project/datasets/test_frames', '/home/larry/ssd_data/sr_project/datasets/test_gt'
    out_dir = '/home/larry/ssd_data/sr_project/experiments/iconvsr/spaceship_test'
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
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
    
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        _ = model(inputs) # 預熱
        start_event.record()
        outputs = model(inputs)
        end_event.record()
        torch.cuda.synchronize()

    runtime_ms = start_event.elapsed_time(end_event)
    ms_per_frame = runtime_ms / inputs.size(1)
    
    print(f"⏱️ 執行速度:")
    print(f"   - 總耗時 ({inputs.size(1)} 幀): {runtime_ms:.2f} ms")
    print(f"   - 單幀耗時: {ms_per_frame:.2f} ms")
    print(f"   - FPS: {1000.0 / ms_per_frame:.2f}")
    
    outputs_np = outputs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    for i, out_img in enumerate(outputs_np):
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), cv2.cvtColor((out_img * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("\n📊 開始計算評估指標...")
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    if gt_paths:
        gts = torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in gt_paths])).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        psnr_fn, ssim_fn, lpips_fn = PeakSignalNoiseRatio(data_range=1.0).to(device), StructuralSimilarityIndexMeasure(data_range=1.0).to(device), lpips.LPIPS(net='alex').to(device)
        
        total_psnr, total_ssim, total_lpips, total_diff = 0, 0, 0, 0
        for i in range(inputs.size(1)):
            out_f, gt_f = outputs[:, i], gts[:, i]
            p, s, l = psnr_fn(out_f, gt_f).item(), ssim_fn(out_f, gt_f).item(), lpips_fn(out_f * 2.0 - 1.0, gt_f * 2.0 - 1.0).item()
            out_std, gt_std = out_f.std().item(), gt_f.std().item()
            diff = abs(out_std - gt_std)
            print(f"Frame {i:02d} | PSNR: {p:.2f} dB, LPIPS: {l:.4f} | Out Std: {out_std:.4f}, GT Std: {gt_std:.4f} (Diff: {diff:.4f})")
            total_psnr += p; total_ssim += s; total_lpips += l; total_diff += diff
        
        print("-" * 40)
        print(f"🏆 序列平均 - PSNR: {total_psnr / inputs.size(1):.2f} dB, SSIM: {total_ssim / inputs.size(1):.4f}, LPIPS: {total_lpips / inputs.size(1):.4f}, Mean Std Diff: {total_diff / inputs.size(1):.4f}")

if __name__ == '__main__':
    main()
