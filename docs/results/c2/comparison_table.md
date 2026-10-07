# C2 Depth Confidence Evaluation

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Std Diff ↓ | Params M | Approx. FLOPs G/frame | Time ms/frame | FPS | Lookahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C1 Renderer MV | 22.84 | 0.7067 | 0.3686 | 0.0045 | 3.42 | N/A | N/A | N/A | 0 frame |
| C2 Renderer MV + Depth confidence | 22.88 | 0.7045 | 0.3741 | 0.0052 | 3.42 | N/A | N/A | N/A | 0 frame |

> THOP may omit DCN and functional confidence operations; FLOPs are incomplete estimates.
> Timing includes confidence, excludes decoding/H2D; Std Diff is spatial, not temporal consistency.
> p95_ms in CSV is mean of per-sequence p95, not pooled global p95.
