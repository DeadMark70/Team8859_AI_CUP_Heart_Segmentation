# AI CUP 2025 電腦斷層心臟肌肉影像分割競賽 - Rank 30 Solution (TEAM_8859)

本儲存庫 (Repository) 包含我們在 **AI CUP 2025 競賽**中的完整訓練與推論流程。我們的最佳成績是基於 **nnU-Net v2** 框架，並採用 **ResEncUNet XL** 架構配合 **5-Fold Ensemble** 策略達成。

## 📋 目錄
* [1. 環境需求與安裝](#1-環境需求與安裝)
* [2. 資料準備與環境變數](#2-資料準備與環境變數)
* [3. 資料前處理](#3-資料前處理)
* [4. 模型訓練](#4-模型訓練)
* [5. 推論與集成](#5-推論與集成)

---

## 1. 環境需求與安裝

### 系統需求
* **OS:** Linux (Ubuntu 20.04+ 推薦)
* **Python:** 3.9+
* **GPU:** 建議使用 NVIDIA V100 或 A100 (訓練 ResEnc XL 需要較大 VRAM)

### 安裝步驟

**1. 建立並啟用虛擬環境 (建議)：**
```bash
conda create -n heart_seg python=3.10
conda activate heart_seg
```

**2. 安裝 Pytorch (請依據您的 CUDA 版本調整)：**
```bash
pip install torch torchvision torchaudio
```

**3. 安裝 nnU-Net v2：**
```bash
pip install nnunetv2
pip install hiddenlayer graphviz  # 選擇性安裝 (用於繪製模型架構)
```

> [!IMPORTANT]
> **⚠️ 重要：替換 Trainer 檔案**
> 為了實現特定的過採樣策略 (**Oversampling 66%**) 與存檔頻率，請務必替換原始套件中的 `nnUNetTrainer.py`。
>
> * **來源檔案：** 本儲存庫中的 `nnUNetTrainer.py`
> * **目標位置：** 您 Python 環境下的 `site-packages/nnunetv2/training/nnUNetTrainer/`
>
> **操作範例：**
> ```bash
> # 假設您在儲存庫根目錄
> cp nnUNetTrainer.py /path/to/your/python/site-packages/nnunetv2/training/nnUNetTrainer/
> ```

---

## 2. 資料準備與環境變數

### 資料夾結構
請依照 nnU-Net 的標準格式整理您的原始資料 (Raw Data)：

```text
nnUNet_raw/
  └── Dataset001_Heart/
      ├── imagesTr/  (訓練集影像)
      ├── labelsTr/  (訓練集標註)
      ├── imagesTs/  (測試集影像 - 上傳預測用)
      └── dataset.json
```

### 設定環境變數
在執行任何指令前，請務必設定以下環境變數（建議寫入 `~/.bashrc`）：

```bash
export nnUNet_raw="/your/path/to/nnUNet_raw"
export nnUNet_preprocessed="/your/path/to/nnUNet_preprocessed"
export nnUNet_results="/your/path/to/nnUNet_results"
```

---

## 3. 資料前處理

本方案使用 **ResEncUNet XL** 架構。請依照以下步驟載入我們的設定檔：

1.  **複製設定檔：**
    請將本儲存庫 `plan/` 資料夾內的以下檔案，複製到您的 `nnUNet_preprocessed/Dataset001_Heart/` 資料夾中（若資料夾不存在請手動建立）：
    * `nnUNetResEncUNetXLPlans.json`
    * `splits_final.json` (確保 5-Fold 切分與我們一致)

2.  **執行預處理指令：**
    ```bash
    nnUNetv2_preprocess -d 001 -c 3d_fullres -p nnUNetResEncUNetXLPlans --verify_dataset_integrity
    ```

---

## 4. 模型訓練

我們對 5 個 Folds 進行了完整訓練。請依序或平行執行以下指令：

* **DATASET_ID:** 001
* **CONFIGURATION:** 3d_fullres
* **TRAINER:** nnUNetTrainer (即步驟 1 替換過的版本)
* **PLANS:** nnUNetResEncUNetXLPlans

```bash
# 範例：訓練 Fold 0 (請根據您的 GPU 數量調整 -num_gpus)
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans -num_gpus 1
```

**重現結果：** 請完成 Fold 0, 1, 2, 3, 4 的訓練。

---

## 5. 推論與集成 (Inference & Ensemble)

我們提供了一個平行化的推論腳本 `batch_ensemble_parallel.py`，可自動執行 TTA (Test Time Augmentation) 並融合 5 個模型。

### ⚠️ 關鍵設定 (執行前必讀！)

在執行腳本之前，請務必打開 `batch_ensemble_parallel.py` 並修改以下路徑，以符合您的本機環境：

```python
# --- batch_ensemble_parallel.py 內的設定區 ---

# 1. 修改測試資料來源路徑
source_imagesTs = "/your/path/to/nnUNet_raw/Dataset001_Heart/imagesTs"

# 2. 修改模型存放路徑 (nnUNet_results 的位置)
model_base_dir = "/your/path/to/nnUNet_results/Dataset001_Heart/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"

# 3. 確認 Checkpoint 名稱 (預設為 checkpoint_best.pth)
checkpoint_map = {
    0: "checkpoint_best.pth",
    1: "checkpoint_best.pth",
    # ...
}
```

### 執行推論

此腳本支援多進程平行處理，適合多顯卡環境。例如，若您有 4 張 GPU，可以同時開啟 4 個終端機執行：

```bash
# 終端機 1 (負責第 1 部分)
python batch_ensemble_parallel.py -p 0 -n 4

# 終端機 2 (負責第 2 部分)
python batch_ensemble_parallel.py -p 1 -n 4

# 終端機 3 (負責第 3 部分)
python batch_ensemble_parallel.py -p 2 -n 4

# 終端機 4 (負責第 4 部分)
python batch_ensemble_parallel.py -p 3 -n 4
```

* `-p` (`--part_id`): 目前的分塊 ID (從 0 開始)
* `-n` (`--num_parts`): 總分塊數

執行完畢後，結果將會自動彙整至腳本中設定的 `final_output_dir`。
