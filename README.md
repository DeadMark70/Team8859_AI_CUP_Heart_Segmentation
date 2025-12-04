# AI CUP 2025 電腦斷層心臟肌肉影像分割競賽 - 第 30 名解決方案 (Rank 30 Solution) TEAM_8859

本儲存庫 (Repository) 包含我們在 AI CUP 2025 競賽中的訓練程式碼與設定檔。我們的最佳成績是基於 nnU-Net v2 框架並採用 ResEncUNet XL 架構訓練而成。

## 1. 環境與安裝 (Environment & Installation)
* **框架 (Framework)**: nnU-Net v2 (based on PyTorch)
* **模型架構 (Model Architecture)**: ResEncUNet XL (`nnUNetResEncUNetXLPlans`)

### 安裝步驟：
1.  安裝標準的 nnU-Net v2。
2.  **重要步驟**：請使用本儲存庫中的 `nnUNetTrainer.py`，**替換**掉您 Python 環境中 nnU-Net 套件原本的 `nnUNetTrainer.py`。
    * **替換位置通常位於**：`.../site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
    * *(說明：我們修改了 Trainer 內的參數以優化訓練結果)*

## 2. 資料前處理與設定檔 (Preprocessing & Plans)
本解決方案使用特定的 **XL 架構** 設定檔，相關檔案位於本儲存庫的 `plan/` 資料夾內。

1.  請將本儲存庫 `plan/` 資料夾中的以下三個檔案，複製到您的 `nnUNet_preprocessed/Dataset001_Heart/` 資料夾中：
    * `nnUNetResEncUNetXLPlans.json` (XL 模型架構定義)
    * `dataset.json` (資料集定義)
    * `splits_final.json` (訓練/驗證集切割定義)

2.  執行標準 nnU-Net 資料前處理 (若您是從原始 Raw Data 開始重現)。

## 3. 訓練 (Training)
請使用修改後的 Trainer 與 XL Plan 執行以下訓練指令：

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans
