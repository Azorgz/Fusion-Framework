import json
import csv
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT_DIR = Path(__file__).parent.parent.parent


class CocoEval:
    MODEL_NAME = "yolo26x.pt"
    DEVICE = "cuda"
    CLASSES = ['person', 'car', 'bicycle', 'bus', 'motorcycle', 'truck']

    def __init__(self, gt_path: str, name: str):
        self.coco_gt = COCO(gt_path)
        self.csv_path = ROOT_DIR / "results/Metrics/detection_results" / f"{name}.csv"
        self.img_dir = None
        self.headers = [
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
        self.model = YOLO(self.MODEL_NAME, task='detect').to(self.DEVICE)

    def __call__(self, img_dir: str | list, method_name: str | list, classes=None):
        if isinstance(img_dir, str):
            assert isinstance(method_name, str)
            self.img_dir = img_dir
            predictions = self.run_inference(classes=classes)
            self.evaluate_coco(predictions, method_name)
        else:
            assert isinstance(method_name, list) and len(img_dir) == len(method_name)
            for d, name in zip(img_dir, method_name):
                self(d, name, classes)

    def save_coco_eval_to_csv(self, coco_eval, method_name):
        stats = coco_eval.stats
        row = [method_name] + [round(x, 5) for x in stats]

        path_exists = os.path.exists(self.csv_path.parent)
        if not path_exists:
            os.makedirs(self.csv_path .parent)

        file_exists = os.path.isfile(self.csv_path)

        # Load existing file if present
        method_found = False
        rows = []
        if file_exists:
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            for i in range(1, len(rows)):
                if rows[i][0] == method_name:
                    rows[i] = row
                    method_found = True
                    break

        # Append if dataset not found
        if not method_found:
            rows.append(row)

        # Ensure header exists
        if not rows or rows[0] != self.headers:
            rows.insert(0, self.headers)

        # Write file
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    @staticmethod
    def _xyxy_to_xywh(box):
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _build_name_to_id_map(json_data):
        return {cls["name"]: cls["id"] for cls in json_data["categories"]}

    def _convert_yolo_to_json_ids(self, yolo_class_dict, json_data):
        name_to_id = self._build_name_to_id_map(json_data)
        mapping = {}

        for yolo_id, class_name in yolo_class_dict.items():
            if class_name in name_to_id:
                mapping[yolo_id] = name_to_id[class_name]
            else:
                pass
        return mapping

    def run_inference(self, classes=None):
        results = []
        mapping = self._convert_yolo_to_json_ids(self.model.names, self.coco_gt.dataset)
        img_id_evaluated = []
        if classes is not None:
            mapping = {yolo_id: json_id for yolo_id, json_id in mapping.items() if self.model.names[yolo_id] in classes}
        else:
            mapping = {yolo_id: json_id for yolo_id, json_id in mapping.items() if self.model.names[yolo_id] in self.CLASSES}

        for img_name in tqdm(sorted(os.listdir(self.img_dir))):
            if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                continue
            img_path = Path(self.img_dir) / img_name
            img_id = int(img_name.split('.')[0].split('_')[-1]) - 1
            preds = self.model(str(img_path), verbose=False, classes=list(mapping.keys()))[0]

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
                    "bbox": self._xyxy_to_xywh(box.tolist()),
                    "score": float(score)
                })
        results.append(img_id_evaluated)
        return results

    def evaluate_coco(self, predictions, method_name):
        # save predictions temporarily
        pred_path = "temp_predictions.json"
        img_id_evaluated = predictions.pop()  # Remove the last element which is img_id_evaluated
        with open(pred_path, "w") as f:
            json.dump(predictions, f)
        coco_dt = self.coco_gt.loadRes(pred_path)
        # Remove the temporary file
        os.remove(pred_path)

        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = img_id_evaluated
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.save_coco_eval_to_csv(coco_eval, method_name)


if __name__ == "__main__":
    csv_path = f"results/Metrics/detection_results/FLIR_seq6.csv"
    dataset = "EXP8"
    methods = ['NightToDay', 'SeAFusion', 'TarDAL', 'TextIF', 'SaliencyMaskedFusion', 'MaeFuse', 'Alpha_blending', 'SAGE', 'PAIF']
    target_gt_json = '/media/godeta/T5 EVO/Datasets/FLIR/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/thermal_annotations.json'
    target_imgs_dir = [f'/home/godeta/PycharmProjects/FusionMethods/results/{m}/{dataset}' for m in methods]
    evaluatorCoco = CocoEval(target_gt_json, 'FLIR_seq6')
    evaluatorCoco(target_imgs_dir, methods)