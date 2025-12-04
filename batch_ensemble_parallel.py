import os
import shutil
import subprocess
import glob
import argparse
import math

# ========================== 設定區域 ==========================

# 1. 原始測試影像資料夾
source_imagesTs = "/home/deadmark70/nnUNet_data/nnUNet_raw/Dataset001_Heart/imagesTs"

# 2. 最終結果輸出路徑 (所有腳本都會存到這裡，不用改)
final_output_dir = "/home/deadmark70/nnUNet_data/nnUNet_results/final_XLensemble_moreacc_submission"

# 3. 模型基礎路徑
model_base_dir = "/home/deadmark70/nnUNet_data/nnUNet_results/Dataset001_Heart/nnUNetTrainer__nnUNetResEncUNetXLPlans__3d_fullres"
dataset_id = "001"
plan_name = "nnUNetResEncUNetXLPlans"
config_name = "3d_fullres"

# 4. Checkpoint 設定
checkpoint_map = {
    0: "checkpoint_best.pth",
    1: "checkpoint_best.pth", 
    2: "checkpoint_best.pth",
    3: "checkpoint_best.pth",
    4: "checkpoint_best.pth"
}

# 5. 安全批次大小 (平行執行時建議設為 1)
# 這樣 4 個腳本同時跑只會佔用約 22GB，保證不爆硬碟
BATCH_SIZE = 1 

# ============================================================

def run_cmd(cmd):
    """執行系統指令"""
    # print(f"  [執行] {cmd}") # 減少噴話以免洗版
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 指令執行失敗: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--part_id", type=int, required=True, help="目前分塊ID (0 ~ num_parts-1)")
    parser.add_argument("-n", "--num_parts", type=int, required=True, help="總分塊數 (例如 4)")
    args = parser.parse_args()

    # 設定這一個腳本只用這張 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.part_id)
    print(f"🚀 啟動工作進程 Part {args.part_id}/{args.num_parts} (使用 GPU {args.part_id})")

    # 準備專屬的臨時工作目錄 (避免衝突)
    work_dir = f"/home/deadmark70/temp_ensemble_work_part_{args.part_id}"
    temp_input = os.path.join(work_dir, "input")
    temp_preds_base = os.path.join(work_dir, "preds")
    temp_ensemble = os.path.join(work_dir, "ensemble_result")

    # 確保最終輸出目錄存在
    os.makedirs(final_output_dir, exist_ok=True)

    # 獲取所有測試檔案
    all_files = sorted(glob.glob(os.path.join(source_imagesTs, "*.nii.gz")))
    total_files = len(all_files)
    
    # 計算這個 Part 要處理哪些檔案
    # 使用 numpy.array_split 的邏輯手動實作
    chunk_size = math.ceil(total_files / args.num_parts)
    start_idx = args.part_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    
    my_files = all_files[start_idx:end_idx]
    print(f"📂 本進程負責處理: {len(my_files)} 個檔案 (Index {start_idx} ~ {end_idx})")

    if len(my_files) == 0:
        print("⚠️ 沒有檔案需要處理，結束。")
        return

    # 開始分批迴圈
    for i in range(0, len(my_files), BATCH_SIZE):
        batch_files = my_files[i : i + BATCH_SIZE]
        print(f"\n[Part {args.part_id}] 🔄 處理批次: {i+1}/{len(my_files)}")

        # 1. 初始化/清理臨時目錄
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(temp_input)
        os.makedirs(temp_ensemble)
        
        fold_out_dirs = []
        for fold in checkpoint_map.keys():
            d = os.path.join(temp_preds_base, f"fold_{fold}")
            os.makedirs(d, exist_ok=True)
            fold_out_dirs.append(d)

        # 2. 複製影像
        for f_path in batch_files:
            shutil.copy(f_path, temp_input)

        # 3. 各 Fold 推論
        for fold, chk_name in checkpoint_map.items():
            output_folder = os.path.join(temp_preds_base, f"fold_{fold}")
            # 注意：這裡不需要 CUDA_VISIBLE_DEVICES，因為已經在環境變數設好了
            cmd = (
                f"nnUNetv2_predict -i {temp_input} -o {output_folder} "
                f"-d {dataset_id} -c {config_name} -f {fold} "
                f"-tr nnUNetTrainer "
                f"-p {plan_name} -chk {chk_name} "
                f"--save_probabilities "
                f"-step_size 0.15 "            
                f"> /dev/null"
            )
            run_cmd(cmd)

        # 4. 集成
        input_folders_str = " ".join(fold_out_dirs)
        # 集成用 CPU 跑即可 (np 設為 2 避免搶資源)
        cmd_ensemble = (
            f"nnUNetv2_ensemble -i {input_folders_str} "
            f"-o {temp_ensemble} -np 2 > /dev/null"
        )
        run_cmd(cmd_ensemble)

        # 5. 移動結果
        generated_files = glob.glob(os.path.join(temp_ensemble, "*.nii.gz"))
        for f in generated_files:
            shutil.move(f, final_output_dir)
            print(f"  [Part {args.part_id}] ✅ 完成: {os.path.basename(f)}")

        # 6. 清理暫存 (釋放硬碟空間)
        shutil.rmtree(work_dir)

    print(f"\n🎉 Part {args.part_id} 全部完成！")

if __name__ == "__main__":
    main()