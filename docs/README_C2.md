# C2：Depth-Aware Motion Confidence / Occlusion Handling

本套程式接續已保存的 `causal_basicvsrpp_mv.py`、`causal_basicvsr_pp.py`、
`train_causal_basicvsrpp_mv.py`、`evaluate_qrisp_c1_mv.py`。
保留 C1 的 causal two-stage、second-order deformable alignment 與 streaming state，
新增使用 Depth 的固定 confidence prior，控制歷史資訊的 DCN modulation mask。

## 方法與本階段範圍

第一版採 encoded-z consistency heuristic，不增加 trainable parameters。
它讓 C2 的差異容易追蹤，C3 才進一步研究可學習的 geometry-conditioned memory，C4 再接 KD。

1. 將 QRISP Depth PNG 解碼成單通道 z-buffer 值。
2. 用當前 renderer MV，取樣前一幀 Depth 到當前座標。
3. 當前／取樣後的 Depth 差異越大，confidence 越低；越界則為零。
4. 一階 confidence 控制 t-1 的 DCN groups；二階 confidence 控制 t-2 的 groups。
5. 在兩個 forward branches 都使用相同 geometry prior。

公式：

```math
\widetilde D_{t-1}(x) = D_{t-1}(x + F_{t\rightarrow t-1}(x))
```

```math
C_t^{(1)}(x) = V_t(x)\exp\left(-|D_t(x)-\widetilde D_{t-1}(x)|/\tau\right)
```

```math
C_t^{(2)}(x)=C_t^{(1)}(x)\cdot
\operatorname{warp}(C_{t-1}^{(1)},F_{t\rightarrow t-1})(x)\cdot V_{t\rightarrow t-2}(x)
```

Depth 在遮擋邊緣使用 nearest sampling，confidence 二階傳遞使用 bilinear。
`gate_strength = α` 時，實際 DCN gate 是 `(1-α)+α*C`。
α=1 完整使用 confidence；α=0 直接執行繼承的 C1 路徑。

**Depth 差異不是精確遮擋標籤。** 當前與過去的 z-buffer 屬於不同 camera view，
相機／物件移動也會使正確 correspondence 的 depth 改變。
本版尚未利用 CameraData 將當前表面投影到 previous view 後比較預期 previous depth。
沒有用 depth 大小直接宣稱 near/far 或 foreground/background，亦未假設 reversed-Z 方向。
因此報告應稱「depth consistency confidence / occlusion-risk handling」。

## Depth 編碼

QRISP supplemental Eq. 4：

```python
depth = R / 255 + G / 255**2 + B / 255**3 + A / 255**4
```

其中 R/G/B/A 是 0–255 的整數。OpenCV 讀入 BGRA，必須重新排序。
不能用 `IMREAD_GRAYSCALE`，不能丟棄 alpha，不能每幀 min-max normalize。
解碼值不是公尺，也未經 linearization。0/1 不自動視為無效，因天空／far-plane 可能使用端點。
極遠區域的 z-buffer 差異可能非常小；固定 tau 對各深度的效果不均勻，需要視覺與 validation 檢查。

資料搭配沿用 C1：

| 輸入 | 路徑 | 模型 shape |
|---|---|---|
| LR RGB | `270p/Native/<angle>/*.png` | `[B,T,3,H,W]` |
| Renderer MV | `270p/MotionVectorsMipBiasMinus2/<angle>/*.exr` | `[B,T,2,H,W]` |
| Depth | `270p/DepthMipBiasMinus2/<angle>/*.png` | `[B,T,1,H,W]` |
| HR GT | `1080p/Native/<angle>/*.png` | `[B,T,3,4H,4W]` |

MV 沿用你已驗證的 OpenCV convention：`dx=-ch2*原始寬度`、`dy=+ch1*原始高度`。
MV 必須先在完整圖上轉 pixel units，再裁切，不能改乘 patch 寬高。
不混入 Jittered modalities，不對時間序列做 reversal。

## 1. 安裝新檔案

先把 zip 放到伺服器的專案目錄；以下假設檔名為 `C2_Depth_Package.zip`。

```bash
conda activate thesis_env
cd /home/larry/ssd_data/sr_project/workspace/thesis_vsr

python -m zipfile -e C2_Depth_Package.zip c2_install

cp c2_install/c2_depth/qrisp_depth.py .
cp c2_install/c2_depth/inspect_qrisp_depth_confidence.py .
cp c2_install/c2_depth/smoke_test_c2_depth.py .
cp c2_install/c2_depth/train_causal_basicvsrpp_mv_depth.py .
cp c2_install/c2_depth/evaluate_qrisp_c2_depth.py .
cp c2_install/c2_depth/archs/basicvsrpp/depth_confidence.py archs/basicvsrpp/
cp c2_install/c2_depth/archs/basicvsrpp/causal_basicvsrpp_mv_depth.py archs/basicvsrpp/
```

這些是新的檔名。需要已有 C1 與其 basicvsrpp 支援模組；評估程式重用
`evaluate_qrisp_c1_mv.py` 的 frame selection 與 metric functions。
若伺服器檔案已自行改版，應先核對 saved C1 的介面。

## 2. 先驗證 Depth 與 confidence

```bash
CUDA_VISIBLE_DEVICES=1 python inspect_qrisp_depth_confidence.py \
  --scene Flooded_Grounds \
  --angle 0013 \
  --pairs 5 \
  --depth-tau 0.001
```

這個已知場景用於格式／視覺檢查；調 tau 時請改用 val_list.txt 裡的 scene/angle，
不要依 test 表現選參數。

預設輸出：

```text
/home/larry/ssd_data/sr_project/experiments/c2_depth_inspection/
```

每個 pair 會保存 RGB、warped RGB、current/previous/warped Depth、confidence、
valid mask、depth delta、RGB error，以及真實 float32 Depth/Confidence `.npy`。
PNG Depth 預覽才有 shared percentile scaling，模型用的 Depth 沒有正規化。
JSON 包含 depth range/std、confidence 分布與條件 RGB warp error。

檢查重點：

- Depth 預覽應顯示合理的場景輪廓；若大致固定且 std=0，先檢查檔案。
- confidence.png 白色表示高可信、黑色表示低可信；valid.png 白色表示有效取樣。
- 越界區應為黑色；深度不連續處若有錯配，應降低 confidence。
- 若幾乎全白，tau 可能過大／z-buffer差異太小；若幾乎全黑，tau 可能過小、格式或配對有誤。
- RGB error 比較只是診斷，不能當作遮擋準確率或有 GT 的證明。

`0.001` 是起始值。可在 val 場景比較 `0.0001 / 0.001 / 0.01`，並保存各 run；
若 gate 太強，可在 val 嘗試 α=0.5。選定後固定設定，再做正式 test。

## 3. 因果性、等價性與反向傳播測試

先測與模型無關的 confidence 計算：

```bash
python smoke_test_c2_depth.py --confidence-only --device cpu
```

再測完整模型：

```bash
CUDA_VISIBLE_DEVICES=1 python smoke_test_c2_depth.py
```

完整測試載入既有 C1 `best_model.pth`，檢查：

- C2 gate-off 與 C1 輸出一致，誤差容許值 1e-6。
- Future Leakage：同時改變未來 RGB/MV/Depth，前綴輸出不受影響。
- Prefix、Clip/Streaming、Chunk continuation 的誤差不超過 1e-6。
- 沒有 SPyNet keys、C1/C2 weights keys 相同。
- SR shape 正確、輸出有限、loss backward 後梯度有限。

預期是 PASS 或接近零的誤差；實際測試結果需在你的 CUDA/MMCV 環境確認。

## 4. 不訓練的診斷比較（validation）

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_qrisp_c2_depth.py \
  --split val \
  --init-from-c1 \
  --depth-tau 0.001 \
  --gate-strength 1 \
  --skip-flops \
  --skip-runtime \
  --output-dir /home/larry/ssd_data/sr_project/experiments/c2_depth_before_finetune
```

兩個模型都使用同一 C1 checkpoint，可先觀察直接加 gate 的影響。
此時畫質下降不代表 C2 無效，因 C1 權重尚未適應 gate；若大幅崩壞，先檢查
confidence 分布、Depth/MV/RGB 配對與 tau，不要直接進長訓練。

## 5. 一個 batch 的資料與訓練 smoke test

```bash
CUDA_VISIBLE_DEVICES=1 python train_causal_basicvsrpp_mv_depth.py \
  --epochs 1 \
  --num-frames 3 \
  --batch-size 1 \
  --num-workers 0 \
  --max-train-batches 1 \
  --max-val-sequences 1 \
  --checkpoint-dir /home/larry/ssd_data/sr_project/checkpoints/c2_depth_smoke
```

確認 Depth 解碼、配對、同步 augmentation、loss backward、validation 與存檔都成功。
smoke checkpoint 與正式 checkpoint 分開。

## 6. 從 C1 fine-tune C2

```bash
CUDA_VISIBLE_DEVICES=1 python train_causal_basicvsrpp_mv_depth.py \
  --epochs 30 \
  --num-frames 15 \
  --patch-size 64 \
  --batch-size 4 \
  --lr 2e-5 \
  --depth-tau 0.001 \
  --gate-strength 1
```

這是第一個 fine-tune 設定，不是保證最適參數。tau/α 用你在 val 選定的值。
保留 C1 的 Charbonnier、Adam、cosine schedule、資料 augmentation 邏輯；
新增 Depth 同步變換並沿用完整圖 MV pixel units。
GPU memory 不足可降低 batch-size，對照組需使用同一設定。

預設載入：

```text
/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv/best_model.pth
```

預設輸出：

```text
/home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv_depth/
  best_model.pth
  latest_checkpoint.pth
```

checkpoint 包含 `confidence_config` 與 `training_protocol`。
resume 必須維持原本 epochs budget、tau、α；例如原本為 30 epochs：

```bash
CUDA_VISIBLE_DEVICES=1 python train_causal_basicvsrpp_mv_depth.py \
  --epochs 30 \
  --depth-tau 0.001 \
  --gate-strength 1 \
  --resume /home/larry/ssd_data/sr_project/checkpoints/causal_basicvsrpp_mv_depth/latest_checkpoint.pth
```

要主張提升來自 Depth，需再做等量 fine-tune control，避免把額外訓練的效果歸因於 gate：

```bash
CUDA_VISIBLE_DEVICES=1 python train_causal_basicvsrpp_mv_depth.py \
  --epochs 30 \
  --num-frames 15 \
  --patch-size 64 \
  --batch-size 4 \
  --lr 2e-5 \
  --depth-tau 0.001 \
  --gate-strength 0 \
  --checkpoint-dir /home/larry/ssd_data/sr_project/checkpoints/c1_matched_finetune
```

α=0 是 C1 路徑，使用同 seed、初始化、資料、epoch budget 與 optimizer。
評估此對照可將 `--c1-checkpoint` 指向該目錄的 `best_model.pth`。

## 7. 正式比較

先一個 sequence smoke：

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_qrisp_c2_depth.py \
  --max-sequences 1 \
  --runtime-repeats 1 \
  --skip-flops \
  --output-dir /home/larry/ssd_data/sr_project/experiments/c2_depth_eval_smoke
```

設定／checkpoint 固定後，再測 test：

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_qrisp_c2_depth.py --skip-flops
```

沿用 C1 discover_sequences 與同一組 metric functions；test 預期 44 sequences × 10 frames。
從 C1 已選定的 RGB frame IDs 尋找 Depth，不重新取四種 modality 的 intersection，
缺 Depth 就報錯。數值 frame IDs 若不連續亦會報錯，避免相鄰 MV 用到跳幀輸入。

輸出 `selection_manifest.json`、`evaluation_config.json`、`per_sequence_results.csv`、
`summary_results.csv`、`comparison_table.md`。
正式 C2 自動讀 checkpoint 保存的 tau/α；CLI override 適用於 validation/ablation，
覆寫值會保存於 evaluation_config。

Latency 是 FP32 streaming CUDA compute，包含 Depth confidence 計算，排除檔案讀取、
PNG 解碼與 H2D，steady-state 排除第一幀。輸出 first-frame 與各序列 p95。
FLOPs 由 THOP 估算，會漏算部分 functional operations 與 DCN，不能據此宣稱 gate 無額外成本。
`Std Diff` 沿用 C1 的空間統計，不能當成 temporal stability 指標。
看 confidence 與運動／新露出區的畫面，另觀察殘影；本工具未計算有 occlusion GT 的評分。

## API 與狀態

```python
from archs.basicvsrpp.causal_basicvsrpp_mv_depth import CausalBasicVSRPlusPlusMVDepth

model = CausalBasicVSRPlusPlusMVDepth(depth_tau=0.001, gate_strength=1).cuda().eval()
# RGB: [B,3,H,W]; motion: [B,2,H,W]; depth: [B,1,H,W]
state = None
with torch.no_grad():
    sr, state = model.forward_step(rgb, motion, depth, state)

# clip: model(rgb_clip, motion_clip, depth_clip)
```

狀態新增 `prev_depth` 與 `prev_confidence_1`；不使用 future Depth、future MV、HR GT 來生成 confidence。
每個新 sequence/scene cut 重設 state=None；同一串流切 chunk 可以保留 state。
訓練 clip 中不 detach recurrent features，維持 BPTT；Depth/confidence prior 以 no_grad 計算。
兩階段共用同一組 confidence，避免重複產生不同定義。

## 本次可驗證範圍

建立環境沒有 PyTorch/CUDA/MMCV/OpenCV，未執行模型、真實 QRISP 或 GPU benchmark。
已執行 Python 語法編譯及 NumPy packed-depth/channel-order/gap-check 測試；
另檢查與 saved C1 的參數名稱／DCN groups 介面。
完整 smoke、訓練與效果仍需伺服器實測，不宣稱畫質已改善或已达到即時速度。

## 參考

- QRISP supplemental Eq. 4：
  https://openaccess.thecvf.com/content/ICCV2023/supplemental/Mercier_Efficient_Neural_Supersampling_ICCV_2023_supplemental.pdf
- PyTorch grid_sample：
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
- BasicVSR++ official reference（C2 實際繼承你的 saved C1）：
  https://github.com/open-mmlab/mmagic/blob/main/mmagic/models/editors/basicvsr_plusplus_net/basicvsr_plusplus_net.py
