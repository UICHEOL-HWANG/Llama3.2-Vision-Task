from datasets import load_dataset, Features, Value, Sequence, Image, DatasetDict
from sklearn.model_selection import train_test_split


features = Features({
    "image_path": Value("string"),
    "query": Value("string"),
    "label": Value("string")
})


dataset = load_dataset("csv", data_files="ocr_dataset.csv", features=features)["train"]


dataset = dataset.cast_column("image_path", Image())


train_test = dataset.train_test_split(test_size=0.1, seed=42)  # 90% / 10%


dataset_dict = DatasetDict({
    "train": train_test["train"],
    "test": train_test["test"]
})


dataset_dict.push_to_hub("UICHEOL-HWANG/Finance-OCR-dataset")
