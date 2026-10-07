# thesis_vsr 資料夾整理

依附圖，根目錄混有 train、run、evaluate、inspect 與 smoke_test。
先依用途分類，C0/C1/C2 保留在原檔名內，避免每個 stage 複製整份架構。

| 位置 | 用途 |
|---|---|
| `archs/basicvsrpp/` | BasicVSR++、C0、C1、C2 架構與 alignment |
| `scripts/train/` | 所有 `train_*.py` |
| `scripts/eval/` | 所有 `evaluate_*.py` |
| `scripts/infer/` | 所有 `run_*.py` |
| `scripts/inspect/` | 所有 `inspect_*.py` |
| `scripts/tests/` | 所有 `smoke_test_*.py` |
| `data_utils/` | `qrisp_depth.py` 等資料讀取 helper |
| 根目錄 `run.py` | 統一執行入口，處理 import 與工作目錄 |
| 根目錄 `organize_project.py` | 整理工具，可重跑搬入新腳本 |
| 根目錄 requirements/environment | 既有環境設定 |
| `backups/folder_cleanup_<timestamp>/` | 搬移前備份與 SHA256 manifest |

資料集、checkpoint、實驗結果維持目前的專案外路徑：

```text
/home/larry/ssd_data/sr_project/datasets/
/home/larry/ssd_data/sr_project/checkpoints/
/home/larry/ssd_data/sr_project/experiments/
```

不搬移 archs、權重、資料集、結果或未知 root helper；不更改任何原始訓練／推論程式內容。
全程不要刪除你目前的 __pycache__，它不影響這次整理。

## 執行

将 `organize_project.py` 和 `run.py` 放到：

```text
/home/larry/ssd_data/sr_project/workspace/thesis_vsr/
```

```bash
conda activate thesis_env
cd /home/larry/ssd_data/sr_project/workspace/thesis_vsr

# 預覽
python organize_project.py

# 搬移、備份、核對原始檔案 bytes
python organize_project.py --apply

# 查看所有指令
python run.py --list
```

遇到同名 destination 會停止，避免覆寫自行修改的版本。
第二次執行可搬新加入根目錄的 C2 腳本；若已全部整理，顯示無檔案需搬移。

## 新執行方式

使用原檔名去掉 `.py`，其餘參數保持原樣：

```bash
CUDA_VISIBLE_DEVICES=1 python run.py evaluate_qrisp_c1_mv --max-sequences 1 --skip-flops --skip-runtime
CUDA_VISIBLE_DEVICES=1 python run.py smoke_test_c1_mv
CUDA_VISIBLE_DEVICES=1 python run.py train_causal_basicvsrpp_mv
CUDA_VISIBLE_DEVICES=1 python run.py inspect_qrisp_motion_vectors --scene Flooded_Grounds --angle 0013
```

launcher 將工作目錄切回專案根目錄，再加入 root、data_utils 與 scripts 各分類到 sys.path。
因此原來的 `from archs...` 及 C2 的 `from evaluate_qrisp_c1_mv...` / `from qrisp_depth...` 可以找到。
後續請使用 run.py 入口，直接執行子資料夾檔案不保證 import 正常。
此工具沒有為未知舊程式的 `__file__` 相對檔案尋址改寫邏輯；如舊腳本用其所在資料夾定位 assets，
需個別核對。訓練前先跑既有 C1 一序列評估與 smoke test。

## C2 檔案放置

| C2 檔案 | 最終位置 |
|---|---|
| `train_causal_basicvsrpp_mv_depth.py` | `scripts/train/` |
| `evaluate_qrisp_c2_depth.py` | `scripts/eval/` |
| `inspect_qrisp_depth_confidence.py` | `scripts/inspect/` |
| `smoke_test_c2_depth.py` | `scripts/tests/` |
| `qrisp_depth.py` | `data_utils/` |
| `depth_confidence.py` | `archs/basicvsrpp/` |
| `causal_basicvsrpp_mv_depth.py` | `archs/basicvsrpp/` |

可以直接複製到上表位置；也可以沿用前次 C2 README 的安裝方式，之後重跑
`python organize_project.py --apply`，便會自動分類根目錄的新 C2 entry scripts/helper。

舊版 C2 smoke test 用所在資料夾查找 `archs/`。整理包的
`c2_compat/smoke_test_c2_depth.py` 已改為向上尋找專案根目錄。
若你已安裝 C2，整理後以此更新版替換 `scripts/tests/smoke_test_c2_depth.py`；
最新的 C2 程式包也已包含此修正。

後續 C2 命令：

```bash
CUDA_VISIBLE_DEVICES=1 python run.py inspect_qrisp_depth_confidence --scene Flooded_Grounds --angle 0013 --pairs 5
python run.py smoke_test_c2_depth --confidence-only --device cpu
CUDA_VISIBLE_DEVICES=1 python run.py smoke_test_c2_depth
```

## 已驗證範圍

已在 temporary fixture 驗證 preview 不變更、apply 搬移與原始 bytes 相同、跨分類 import、
CLI arguments、工作目錄、重跑與同名衝突保護。
未連線到你的伺服器，未執行真實 C1 模型。
