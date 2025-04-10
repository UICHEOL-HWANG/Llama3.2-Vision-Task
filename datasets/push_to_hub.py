from datasets import load_dataset, Features, Value, Sequence, Image, DatasetDict
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from PIL import Image

def resize_images_in_csv(csv_path, target_dir, size=(560, 560)):


    df = pd.read_csv(csv_path)
    os.makedirs(target_dir, exist_ok=True)

    new_paths = []

    for _, row in df.iterrows():

        orig_path = row["image_path"]
        try:
            img = Image.open(orig_path).convert("RGB")

            # 가로로 누워있으면 세로로 회전
            if img.width > img.height:
                img = img.rotate(270, expand=True)

            img = img.resize(size, Image.LANCZOS)

            new_path = os.path.join(target_dir, os.path.basename(orig_path))
            img.save(new_path)
            new_paths.append(new_path)
        except Exception as e:
            print(f"❌ 이미지 오류: {orig_path} - {e}")
            new_paths.append(orig_path)  # 실패 시 기존 경로 유지

    # 결과 DataFrame 생성
    df_subset = df.iloc[:len(new_paths)].copy()
    df_subset["image_path"] = new_paths
    df_subset.to_csv(csv_path.replace(".csv", "_10k.csv"), index=False)

resize_images_in_csv("./csv/ocr_dataset.csv", "resized_images")

features = Features({
    "image_path": Value("string"),
    "query": Value("string"),
    "label": Value("string")
})


dataset = load_dataset("csv", data_files="./csv/ocr_dataset_10k.csv", features=features)["train"]
dataset = dataset.cast_column("image_path", Image())
train_test = dataset.train_test_split(test_size=0.1, seed=42)  # 90% / 10%


dataset_dict = DatasetDict({
    "train": train_test["train"],
    "test": train_test["test"]
})


dataset_dict.push_to_hub("UICHEOL-HWANG/Finance-OCR-dataset")
