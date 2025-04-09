import os
import csv
import json

def extract_text_sorted_by_position(annotations):
    polygons = annotations[0]['polygons']
    sorted_polygons = sorted(
        polygons,
        key=lambda x: (min(p[1] for p in x['points']), min(p[0] for p in x['points']))
    )
    return [p['text'] for p in sorted_polygons]  # 리스트 형태 유지

def generate_csv(annotation_dir, image_dir, output_csv="ocr_dataset.csv"):
    with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "query", "label"])
        writer.writeheader()

        for filename in os.listdir(annotation_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(annotation_dir, filename), encoding='utf-8') as jf:
                        data = json.load(jf)

                    image_filename = data["name"]
                    image_path = os.path.join(image_dir, image_filename)
                    if not os.path.exists(image_path):
                        continue

                    text_list = extract_text_sorted_by_position(data["annotations"])
                    instruction = "이미지를 기반으로 금융 문서의 텍스트를 순서대로 인식하고, 문서의 구조를 고려하여 정확히 추출하세요."

                    writer.writerow({
                        "image_path": image_path,
                        "query": instruction,
                        "label": json.dumps(text_list, ensure_ascii=False)
                    })

                except Exception as e:
                    print(f"❌ 에러: {filename} - {e}")

if __name__ == "__main__":
    generate_csv(
        annotation_dir="./result/bank/annotations",
        image_dir="./result/bank/images",
        output_csv="ocr_dataset.csv"
    )
