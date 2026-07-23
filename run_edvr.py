import os
import torch
import cv2
import glob
import numpy as np
from archs.edvr.edvr import EDVRNet
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from thop import profile

def main():
    print("🔄 正在初始化 EDVR (純淨 Bare-Metal 模式)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EDVRNet(num_frames=5, with_tsa=True).to(device)
    
    ckpt_path = '/home/larry/ssd_data/sr_project/checkpoints/edvr/best_model.pth'
    if os.path.exists(ckpt_path):
        clean_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(ckpt_path, map_location=device).get('state_dict', torch.load(ckpt_path)).items() if k not in ['step_counter', 'meta']}
        model.load_state_dict(clean_state_dict, strict=True)
        print("✅ 成功載入 best_model.pth")
    else:
        print("⚠️ 找不到 best_model.pth，請先執行訓練。")
        return
        
    model.eval()

    input_dir, gt_dir = '/home/larry/ssd_data/sr_project/datasets/test_frames', '/home/larry/ssd_data/sr_project/datasets/test_gt'
    out_dir = '/home/larry/ssd_data/sr_project/experiments/edvr/spaceship_test'
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in img_paths]
    
    # 邊界補齊 (Padding) 給滑動視窗使用 (長度 = 5, padding = 2)
    imgs_padded = [imgs[0]] * 2 + imgs + [imgs[-1]] * 2
    
    print(f"🚀 執行滑動視窗推論 (總幀數: {len(imgs)})...")
    
    dummy_input = torch.zeros(1, 5, 3, imgs[0].shape[0], imgs[0].shape[1]).to(device)
    macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"⚡ 單幀運算量 (FLOPs/Frame): {(macs * 2) / 1e9:.2f} G")
    
    outputs_list = []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        _ = model(dummy_input) # 預熱
        start_event.record()
        for i in range(len(imgs)):
            window = imgs_padded[i : i + 5]
            input_tensor = torch.from_numpy(np.stack(window)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
            out_frame = model(input_tensor) # EDVR 輸出是 [1, 3, 4H, 4W]
            outputs_list.append(out_frame)
        end_event.record()
        torch.cuda.synchronize()

    outputs = torch.stack(outputs_list, dim=1) # 合併成 [1, 10, 3, 1080, 1920]
    runtime_ms = start_event.elapsed_time(end_event)
    print(f"⏱️ 執行速度: 總耗時 {runtime_ms:.2f} ms | FPS: {1000.0 / (runtime_ms / len(imgs)):.2f}")
    
    outputs_np = outputs.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    for i, out_img in enumerate(outputs_np):
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), cv2.cvtColor((out_img * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("\n📊 開始計算評估指標...")
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    if gt_paths:
        gts = torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for p in gt_paths])).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        psnr_fn, ssim_fn, lpips_fn = PeakSignalNoiseRatio(data_range=1.0).to(device), StructuralSimilarityIndexMeasure(data_range=1.0).to(device), lpips.LPIPS(net='alex').to(device)
        
        total_psnr, total_ssim, total_lpips, total_std = 0, 0, 0, 0
        for i in range(len(imgs)):
            out_f, gt_f = outputs[:, i], gts[:, i]
            p, s, l, st = psnr_fn(out_f, gt_f).item(), ssim_fn(out_f, gt_f).item(), lpips_fn(out_f * 2.0 - 1.0, gt_f * 2.0 - 1.0).item(), out_f.std().item()
            print(f"Frame {i:02d} | PSNR: {p:.2f} dB, SSIM: {s:.4f}, LPIPS: {l:.4f}, Std: {st:.4f}")
            total_psnr += p; total_ssim += s; total_lpips += l; total_std += st
        
        print("-" * 40)
        print(f"🏆 序列平均 - PSNR: {total_psnr / len(imgs):.2f} dB, SSIM: {total_ssim / len(imgs):.4f}, LPIPS: {total_lpips / len(imgs):.4f}, Std: {total_std / len(imgs):.4f}")

if __name__ == '__main__':
    main()
