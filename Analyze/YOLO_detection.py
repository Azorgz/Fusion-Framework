import importlib
import json
import csv
import os
import time
from pathlib import Path
from tqdm import tqdm
import re

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from fusion_framework.datasets import DATASETS, get_dataset_class
from fusion_framework.options.options import Options

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
    # Ensure stats exist before rounding (handles empty/failed evaluations gracefully)
    stats = coco_eval.stats if coco_eval.stats is not None else [0]*12
    new_row = [method_name] + [round(x, 5) for x in stats]

    csv_path = Path(os.getcwd()).parent / csv_path

    # Ensure the parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    method_updated = False
    headers = ["method"] + ['AP@[0.50:0.95]', 'AP@0.50', 'AP@0.75', 'AP_small', 'AP_medium',
                            'AP_large', 'AR@1', 'AR@10', 'AR@100', 'AR_small', 'AR_medium', 'AR_large']
    # Check if file exists and read its current contents
    if csv_path.is_file():
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
                all_rows.append(headers)
            except StopIteration:
                pass  # File is completely empty

            # Loop through existing data
            for row in reader:
                if row and row[0] == method_name:
                    all_rows.append(new_row)  # Replace with updated metrics
                    method_updated = True
                else:
                    all_rows.append(row)
    else:
        # If the file doesn't exist, start with headers
        all_rows.append(headers)

    # If the method wasn't found to be updated, append it as a new row
    if not method_updated:
        all_rows.append(new_row)

    # WRITE the data back to the CSV file
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def build_name_to_id_map(json_data):
    dic = {cls["name"].lower(): cls["id"] for cls in json_data["categories"]}
    return {d.replace("people", "person"): id for d, id in dic.items()}


def convert_yolo_to_json_ids(yolo_class_dict, json_data):
    name_to_id = build_name_to_id_map(json_data)

    mapping = {}

    for yolo_id, class_name in yolo_class_dict.items():
        if class_name in name_to_id:
            mapping[yolo_id] = name_to_id[class_name]
        else:
            pass

    return mapping


def build_filename_to_id_map(coco_gt):
    """
    Creates a mapping from the extracted numerical digits of a filename to its COCO image ID.
    Example: 'FLIR_01234.jpg' -> digits '01234' -> int(1234) -> image_id
    """
    filename_to_id = {}

    for img_id, img_info in coco_gt.imgs.items():
        filename = img_info['file_name']
        # Extract all continuous digits from the ground-truth filename
        match = re.search(r'\d+', os.path.basename(filename))
        if match:
            # Convert to int to strip leading zeros, ensuring consistency
            digit_key = int(match.group())
            filename_to_id[digit_key] = img_id

    return filename_to_id


def run_inference(model, image_dir, coco_gt, classes=None):
    results = []
    mapping = convert_yolo_to_json_ids(model.names, coco_gt.dataset)
    img_id_evaluated = []
    if classes is not None:
        mapping = {yolo_id: json_id for yolo_id, json_id in mapping.items() if model.names[yolo_id] in classes}

        # 1. Build the high-speed ID mapping dictionary
    filename_to_id = build_filename_to_id_map(coco_gt)

    for img_name in tqdm(sorted(os.listdir(image_dir)) if isinstance(image_dir, str) else image_dir):
        if not img_name.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            continue
        if isinstance(image_dir, str):
            img_path = Path(image_dir) / img_name
        else:
            img_path = img_name

        # 2. Extract digits from your target image name
        match = re.search(r'\d+', img_name.split('/')[-1])  # Extract digits from the filename
        if not match:
            print(f"Warning: No digits found in target filename {img_name}, skipping.")
            continue

        digit_key = int(match.group())

        # 3. Safely look up the matching COCO ID
        if digit_key in filename_to_id:
            img_id = filename_to_id[digit_key]
        else:
            print(f"Warning: Could not match digits {digit_key} from {img_name} to coco_gt. Skipping.")
            continue

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
    # Save predictions temporarily
    pred_path = "temp_predictions.json"
    img_id_evaluated = predictions.pop()  # Remove the last element which is img_id_evaluated

    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    coco_dt = coco_gt.loadRes(pred_path)

    # Filter image IDs by time_of_day metadata
    day_img_ids = []
    night_img_ids = []

    for img_id in img_id_evaluated:
        img_info = coco_gt.imgs.get(img_id, {})
        time_of_day = img_info.get("time_of_day", "").lower()

        if "day" in time_of_day:
            day_img_ids.append(img_id)
        elif "night" in time_of_day:
            night_img_ids.append(img_id)

    # FIXED: Define the evaluation splits OUTSIDE the loop
    eval_splits = [
        {"suffix": "_combined", "ids": img_id_evaluated},
        {"suffix": "_day", "ids": day_img_ids},
        {"suffix": "_night", "ids": night_img_ids}
    ]

    # Run COCO evaluation for each split
    for split in eval_splits:
        if not split["ids"]:
            print(f"Warning: No images found for split {split['suffix']}. Skipping.")
            continue

        print(f"Evaluating {method_name} on split: {split['suffix'].replace('_', '')} with {len(split['ids'])} images.")

        # FIXED: Instantiate a fresh instance for every single split to ensure clean states
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = split["ids"]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Save to CSV using a unique method name per split (e.g., 'yolov8_day')
        split_method_name = f"{method_name}{split['suffix']}"
        save_coco_eval_to_csv(coco_eval, split_method_name, csv_path)


def main(coco_gt, target_imgs_dir, method_name, csv_path):

    # load model
    model = YOLO(MODEL_NAME, task='detect')

    # run inference & evaluate for IR images
    predictions = run_inference(model, target_imgs_dir, coco_gt, classes=['person', 'car', 'bicycle', 'bus', 'motorcycle', 'truck']) #
    evaluate_coco(coco_gt, predictions, method_name, csv_path)


def train(path_to_yaml):
    model = YOLO(MODEL_NAME, task='detect')
    model.train(data=path_to_yaml, epochs=2, imgsz=640, batch=8, device=DEVICE)


if __name__ == "__main__":
    task = 'detect' # or 'detect'
    dataset = 'FLIR_NIGHT'

    if task == 'detect':
        method_names = ['Visible', 'Infrared', 'Alpha_blending', 'SeAFusion', 'TarDAL', 'TextIF', 'MaeFuse', 'NightToDay']
        for method_name in method_names:
            csv_path = f"results/Metrics/detection_results/{dataset}.csv"
            if dataset == 'LYNRED_DETECTION_test' or dataset == 'LYNRED_DETECTION_val' or dataset == 'IGNITE':
                target_gt_json = '/home/godeta/Téléchargements/LYNRED_multimodal_detection_V1/detection_dataset/metadata/vis_test.json'
                imgs_metadata = '/home/godeta/Téléchargements/LYNRED_multimodal_detection_V1/detection_dataset/metadata/ir_test.json'
                # load ground truth
                coco_gt = COCO(target_gt_json)
                coco_meta = COCO(imgs_metadata)
                coco_gt.imgs.update(coco_meta.imgs)  # Merge the image metadata from both JSON files
                target_imgs_dir = f'/home/godeta/PycharmProjects/FusionMethods/results/{method_name}/{dataset}/'
            elif dataset == 'FLIR_aligned_val':
                target_gt_json = '/media/godeta/T5 EVO/Datasets/FLIR/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/val/thermal_annotations.json'
                coco_gt = COCO(target_gt_json)
                target_imgs_dir = f'/home/godeta/PycharmProjects/FusionMethods/results/CrossRAFT_{method_name}/FLIR_aligned_val'
            elif dataset == 'FLIR_aligned_train':
                target_gt_json = '/media/godeta/T5 EVO/Datasets/FLIR/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/thermal_annotations.json'
                coco_gt = COCO(target_gt_json)
                target_imgs_dir = f'/home/godeta/PycharmProjects/FusionMethods/results/CrossRAFT_{method_name}/FLIR_aligned_train'
            elif dataset == 'FLIR_NIGHT':
                target_gt_json = '/media/godeta/T5 EVO/Datasets/FLIR/FLIR_ADAS_1_3_full/FLIR_ADAS_1_3_train/train/thermal_annotations.json'
                coco_gt = COCO(target_gt_json)
                target_imgs_dir = f'/home/godeta/PycharmProjects/FusionMethods/results/{method_name}/FLIR_NIGHT'
            elif dataset == 'M3FD_detection':
                target_gt_json = '/media/godeta/T5 EVO/Datasets/M3FD/Detection/metadata.json'
                coco_gt = COCO(target_gt_json)
                target_imgs_dir = f'/home/godeta/PycharmProjects/FusionMethods/results/{method_name}/M3FD_detection/'
            if method_name == 'Visible' or method_name == 'Infrared':
                opt = Options().parse()
                module = importlib.import_module(f"{DATASETS[dataset.lower()]}", package='fusion_framework.datasets')
                dataset_cls = get_dataset_class(module)
                dataset_instance = dataset_cls(opt)
                if method_name == 'Visible':
                    target_imgs_dir = dataset_instance.image_vis
                else:
                    target_imgs_dir = dataset_instance.image_ir

            main(coco_gt, target_imgs_dir, method_name, csv_path)

    elif task == 'profile':
        import pandas as pd

        # def generate_latex_table(csv_path, output_tex_path):
        #     # Load data
        #     df = pd.read_csv(csv_path)
        #
        #     # 1. Rename the methods as requested
        #     df['method'] = df['method'].replace({
        #         'IGNITE': 'IGNITE (ours)',
        #         'Alpha_blending': 'Alpha 0.5'
        #     })
        #
        #     # 2. Establish the exact desired row layout
        #     desired_prefix = ['Visible', 'Infrared', 'Alpha 0.5']
        #     middle_methods = [m for m in df['method'] if m not in desired_prefix and m != 'IGNITE (ours)']
        #     final_order = desired_prefix + middle_methods + ['IGNITE (ours)']
        #
        #     # Reindex the DataFrame to match this layout
        #     df = df.set_index('method').loc[final_order].reset_index()
        #
        #     # 3. Clean COCO metric mapping for LaTeX headers
        #     header_mapping = {
        #         'method': 'Method',
        #         'AP@[0.50:0.95]': r'AP',
        #         'AP@0.50': r'AP$_{50}$',
        #         'AP@0.75': r'AP$_{75}$',
        #         # 'AP_small': r'AP$_S$',
        #         # 'AP_medium': r'AP$_M$',
        #         # 'AP_large': r'AP$_L$',
        #         'AR@1': r'AR$_1$',
        #         'AR@10': r'AR$_{10}$',
        #         'AR@100': r'AR$_{100}$',
        #         # 'AR_small': r'AR$_S$',
        #         # 'AR_medium': r'AR$_M$',
        #         # 'AR_large': r'AR$_L$'
        #     }
        #
        #     numeric_cols = [c for c in df.columns if c in list(header_mapping.keys()) and c != 'method']
        #
        #     # Find column-wise maximums to automatically apply bold formatting
        #     max_values = df[numeric_cols].max()
        #
        #     # Generate the LaTeX string layout
        #     latex_lines = []
        #     latex_lines.append(r'\begin{table*}[t]')
        #     latex_lines.append(r'\centering')
        #     latex_lines.append(r'\small')
        #     latex_lines.append(r'\setlength{\tabcolsep}{2pt}')
        #     latex_lines.append(r'\caption{Object detection performance comparison on the LYNRED dataset.}')
        #     latex_lines.append(r'\label{tab:lynred_detection}')
        #
        #     # Alignment string: left-aligned for names, centered for numeric metrics
        #     align = 'l' + 'c' * len(numeric_cols)
        #     latex_lines.append(f'\\begin{{tabular}}{{{align}}}')
        #     latex_lines.append(r'\toprule')
        #
        #     # Append Header Row
        #     headers = [header_mapping.get(c, c) for c in df.columns if c not in ['AP_small', 'AP_medium', 'AP_large', 'AR_small', 'AR_medium', 'AR_large']]
        #     latex_lines.append(' & '.join(headers) + r' \\')
        #     latex_lines.append(r'\midrule')
        #
        #     # Append Data Rows
        #     for idx, row in df.iterrows():
        #         row_str = []
        #         method_name = row['method']
        #
        #         # Apply bold typeface to your method label
        #         if method_name == 'IGNITE (ours)':
        #             row_str.append(r'\textbf{IGNITE (ours)}')
        #         else:
        #             row_str.append(method_name)
        #
        #         for col in numeric_cols:
        #             val = row[col]
        #             val_str = f'{val:.3f}'  # Rounds beautifully to 3 decimals
        #
        #             # If value is the column max, wrap in \mathbf
        #             if val == max_values[col]:
        #                 val_str = f'\\mathbf{{{val_str}}}'
        #
        #             row_str.append(f'${val_str}$')
        #
        #         # Add a visual separation line right before 'ours' to highlight it
        #         if method_name == 'IGNITE (ours)':
        #             latex_lines.append(r'\midrule')
        #
        #         latex_lines.append(' & '.join(row_str) + r' \\')
        #
        #     latex_lines.append(r'\bottomrule')
        #     latex_lines.append(r'\end{tabular}')
        #     latex_lines.append(r'\end{table*}')
        #
        #     # Compile and save
        #     latex_code = '\n'.join(latex_lines)
        #     with open(output_tex_path, 'w') as f:
        #         f.write(latex_code)
        #
        #     print("LaTeX code successfully generated and saved to:", output_tex_path)
        #     return latex_code
        import pandas as pd
        import numpy as np


        def generate_custom_latex_table(csv_paths, output_tex_path, table_label="tab:multi_dataset_detection"):
            # Dataset keyword to clean display name mapping
            dataset_mapping = {
                'LYNRED_DETECTION_test': 'Lynred Detection',
                'M3FD_detection': 'M3FD',
                'FLIR_NIGHT': 'FLIR'
            }

            # Method display name cleanup mapping
            method_rename = {
                'NightToDay': 'IGNITE (ours)',
                'Alpha_blending': 'Alpha 0.5',
                'Visible': 'Visible',
                'Infrared': 'Infrared',
                'SeAFusion': 'SeAFusion',
                'TarDAL': 'TarDAL',
                'TextIF': 'TextIF',
                'MaeFuse': 'MaeFuse'
            }

            # Metric mappings to LaTeX subscripts
            header_mapping = {
                'AP@[0.50:0.95]': r'AP\footnotesize$_{50:95}$',
                'AP@0.50': r'AP$_{50}$',
                'AR@100': r'AR$_{100}$'
            }
            metrics = list(header_mapping.keys())

            # Desired logical column order for datasets
            datasets_order = ['Lynred Detection', 'M3FD', 'FLIR']

            # Store processed dataframes for each dataset
            combined_data = {}

            for path in csv_paths:
                # Detect dataset name using keywords anywhere in the filepath/filename
                display_name = None
                for key, val in dataset_mapping.items():
                    if key in os.path.basename(path) or key in path:
                        display_name = val
                        break

                # Fallback to filename base if no mapping keyword matches
                if display_name is None:
                    display_name = os.path.splitext(os.path.basename(path))[0]

                df = pd.read_csv(path)

                # 1. Filter rows to consider only the '_combined' results
                df_combined = df[df['method'].str.endswith('_combined')].copy()

                # 2. Extract base method name and map to clean display titles
                df_combined['base_method'] = df_combined['method'].apply(lambda x: x.replace('_combined', ''))
                df_combined['base_method'] = df_combined['base_method'].map(method_rename).fillna(
                    df_combined['base_method'])

                # 3. Restructure to use base_method as index and slice the required metrics
                df_combined = df_combined.set_index('base_method')[metrics]

                combined_data[display_name] = df_combined

            # Order rows as established in previous layouts
            row_order = ['Visible', 'Infrared', 'Alpha 0.5', 'SeAFusion', 'TarDAL', 'TextIF', 'MaeFuse',
                         'IGNITE (ours)']

            # Create an empty MultiIndex DataFrame for aligning all datasets together
            multi_cols = pd.MultiIndex.from_product([datasets_order, metrics], names=['dataset', 'metric'])
            df_pivot = pd.DataFrame(index=row_order, columns=multi_cols, dtype=float)

            for d in datasets_order:
                for m in metrics:
                    if d in combined_data and m in combined_data[d].columns:
                        # Use reindex to safely align methods across datasets even if some are missing
                        df_pivot[(d, m)] = combined_data[d][m].reindex(row_order)

            # Calculate column-wise maximums for highlighting best results
            max_values = df_pivot.max()

            # 4. Generate the LaTeX Table Structure
            latex_lines = []
            latex_lines.append(r'\begin{table*}[t]')
            latex_lines.append(r'\centering')
            latex_lines.append(r'\small')
            latex_lines.append(r'\setlength{\tabcolsep}{5pt}')
            latex_lines.append(
                r'\caption{Object detection performance comparison across different datasets (combined results).}')
            latex_lines.append(f'\\label{{{table_label}}}')

            # Column setups (1 method column + 3 metrics per dataset * 3 datasets = 10 columns total)
            align = 'l' + (' | ' + 'c' * len(metrics)) * len(datasets_order)
            latex_lines.append(f'\\begin{{tabular}}{{{align}}}')
            latex_lines.append(r'\toprule')

            # Top Header Layer (Methods spans 2 rows, Datasets span 3 columns each)
            top_header = [r'\multirow{2}{*}{\textbf{Methods}}']
            for i, d in enumerate(datasets_order):
                pipe = '|' if i < len(datasets_order) - 1 else ''
                top_header.append(f'\\multicolumn{{{len(metrics)}}}{{c{pipe}}}{{\\textbf{{{d}}}}}')
            latex_lines.append(' & '.join(top_header) + r' \\')

            # Partial horizontal dividing rule from column 2 to 10
            latex_lines.append(f'\\cline{{2-{1 + len(datasets_order) * len(metrics)}}}')

            # Sub-Header Layer (AP, AP50, AR100)
            sub_headers = ['']
            for d in datasets_order:
                for m in metrics:
                    sub_headers.append(header_mapping[m])
            latex_lines.append(' & '.join(sub_headers) + r' \\')
            latex_lines.append(r'\midrule')

            # 5. Build Row Elements
            for method in row_order:
                row_str = []
                if method == 'IGNITE (ours)':
                    latex_lines.append(r'\midrule')
                    row_str.append(r'\textbf{IGNITE (ours)}')
                else:
                    row_str.append(method)

                for d in datasets_order:
                    for m in metrics:
                        val = df_pivot.loc[method, (d, m)]
                        if pd.isna(val):
                            row_str.append('---')  # Visual missing-value placeholder
                        else:
                            val_str = f'{val:.3f}'
                            # Bold column maximum values
                            if val == max_values[(d, m)]:
                                val_str = f'\\mathbf{{{val_str}}}'
                            row_str.append(f'${val_str}$')

                latex_lines.append(' & '.join(row_str) + r' \\')

            latex_lines.append(r'\bottomrule')
            latex_lines.append(r'\end{tabular}')
            latex_lines.append(r'\end{table*}')

            # Save output to disk
            latex_code = '\n'.join(latex_lines)
            with open(output_tex_path, 'w') as f:
                f.write(latex_code)

            return latex_code

        # Run the generation script
        path = [f'/home/godeta/PycharmProjects/FusionMethods/results/Metrics/detection_results/LYNRED_DETECTION_test.csv',
                f'/home/godeta/PycharmProjects/FusionMethods/results/Metrics/detection_results/M3FD_detection.csv',
                f'/home/godeta/PycharmProjects/FusionMethods/results/Metrics/detection_results/FLIR_NIGHT.csv']
        latex_table = generate_custom_latex_table(path, 'detection_table.tex')

    elif task == 'train':
        train()