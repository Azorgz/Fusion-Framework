import json
import csv
import os
from pathlib import Path
from tqdm import tqdm

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -------------------------------------------------
# Configuration
# -------------------------------------------------
VIDEO_PATH = "/home/godeta/PycharmProjects/FusionMethods/results/videos/intro2.mp4"
OUTPUT_PATH = "/home/godeta/PycharmProjects/FusionMethods/results/videos/intro_detected.mp4"
MODEL_NAME = "yolo26x.pt"
DEVICE = "cuda"

# -------------------------------------------------
# Load Model
# -------------------------------------------------
# model.predict(
#     source=VIDEO_PATH,
#     save=True,
#     classes=list(range(1)),
#     conf=0.6)

headers = [
    "method",
    "AP@[0.50:0.95]",
    "AP@0.50",
    "AP@0.75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def save_coco_eval_to_csv(coco_eval, method_name, csv_path):
    row = [method_name] + [round(x, 5) for x in coco_eval.stats]

    csv_path = Path(os.getcwd()).parent / csv_path

    path_exists = os.path.exists(csv_path.parent)
    if not path_exists:
        os.makedirs(csv_path.parent)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # write header only once
        if not file_exists:
            writer.writerow(headers)

        writer.writerow(row)

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def build_name_to_id_map(json_data):
    return {cls["name"]: cls["id"] for cls in json_data["categories"]}


def convert_yolo_to_json_ids(yolo_class_dict, json_data):
    name_to_id = build_name_to_id_map(json_data)

    mapping = {}

    for yolo_id, class_name in yolo_class_dict.items():
        if class_name in name_to_id:
            mapping[yolo_id] = name_to_id[class_name]
        else:
            pass

    return mapping


def run_inference(model, image_dir, coco_gt, classes=None):
    results = []
    mapping = convert_yolo_to_json_ids(model.names, coco_gt.dataset)
    img_id_evaluated = []
    if classes is not None:
        mapping = {yolo_id: json_id for yolo_id, json_id in mapping.items() if model.names[yolo_id] in classes}

    for img_name in tqdm(sorted(os.listdir(image_dir))):
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            continue
        img_path = Path(image_dir) / img_name
        img_id = int(img_name.split('.')[0].split('_')[-1]) - 1  # Assuming img_name format is like 'frame_000001.jpg'
        preds = model(str(img_path), verbose=False, classes=list(mapping.keys()))[0]

        if preds.boxes is None:
            continue

        boxes = preds.boxes.xyxy.cpu().numpy()
        scores = preds.boxes.conf.cpu().numpy()
        classes = preds.boxes.cls.cpu().numpy().astype(int)
        img_id_evaluated.append(img_id)

        for box, score, cls in zip(boxes, scores, classes):
            results.append({
                "image_id": img_id,
                "category_id": mapping[int(cls)],
                "bbox": xyxy_to_xywh(box.tolist()),
                "score": float(score)
            })
    results.append(img_id_evaluated)
    return results


def evaluate_coco(coco_gt, predictions, method_name, csv_path):

    # save predictions temporarily
    pred_path = "temp_predictions.json"
    img_id_evaluated = predictions.pop()  # Remove the last element which is img_id_evaluated
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = img_id_evaluated
    coco_eval.evaluate()
    coco_eval.accumulate()
    save_coco_eval_to_csv(coco_eval, method_name, csv_path)


def main(target_gt_json, target_imgs_dir, method_name, csv_path):

    # load model
    model = YOLO(MODEL_NAME, task='detect')

    # load ground truth
    coco_gt = COCO(target_gt_json)

    # run inference & evaluate for IR images
    predictions = run_inference(model, target_imgs_dir, coco_gt, classes=['person', 'car', 'bicycle', 'bus', 'motorcycle', 'truck'])
    evaluate_coco(coco_gt, predictions, method_name, csv_path)


if __name__ == "__main__":
    csv_path = f"results/Metrics/detection_results/FLIR_seq6.csv"
    dataset = "EXP6"
    method_name = "vis"
    target_gt_json = '/media/godeta/T5 EVO/Datasets/FLIR/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/thermal_annotations.json'
    target_imgs_dir = '/home/godeta/Bureau/selection sequence/seq6/vis/'
    main(target_gt_json, target_imgs_dir, method_name, csv_path)