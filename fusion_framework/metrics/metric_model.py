import csv
import warnings
from pathlib import Path
from warnings import warn

import numpy as np
import torch
from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import METRICS_DICT
from torch import isfinite


class MetricModel:
    def __init__(self, metrics: [str], device: torch.device, path: Path | str):
        self.metrics = {}
        self.device = device
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
        if metrics == 'all':
            metrics = list(METRICS_DICT.keys())
        for m in metrics:
            if m.lower() in METRICS_DICT:
                metric = METRICS_DICT[m.lower()](device)
                self.metrics[m] = metric
            else:
                warn(f'The metric {m} doesnt exist. Please choose a metric in {METRICS_DICT.keys()}')
        self._init_results()

    def _init_results(self):
        self.results = {k: 0 for k, v in self.metrics.items()}
        self.count = {k: 0 for k, v in self.metrics.items()}

    def _reinit_metrics(self):
        self.metrics = {n: METRICS_DICT[n.lower()](self.device) for n in self.metrics.keys()}

    def _compute_metrics(self, ir: torch.tensor, vis: torch.tensor, fus: torch.tensor):
        name = ""
        for name, metric in self.metrics.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if metric.max_arg == 3:
                    result = metric(ir, vis, fus)
                elif metric.max_arg == 2:
                    result = metric(ir, vis)
                else:
                    result = metric(fus)
                if result is not None and isfinite(result) and not torch.isnan(result):
                    self.results[name] += result.detach().cpu().numpy()
                    self.count[name] += 1
        torch.cuda.empty_cache()
        if self.count[name] % 200 == 0:
            self._reinit_metrics()

    def __call__(self, vis, ir, fus):
        if len(self.metrics) == 0:
            return
        fus_img = ImageTensor(fus).RGB().to(self.device) if fus is not None else None
        ir_img = ImageTensor(ir).GRAY().match_shape(fus_img).to(self.device) if ir is not None else None
        vis_img = ImageTensor(vis).match_shape(fus_img).to(self.device) if vis is not None else None
        self._compute_metrics(ir_img, vis_img, fus_img)

    def save_results(self, method: str, dataset: str):
        """
        Save the results in a CSV. If the CSV already exist, a new line will be added with the new results.
        If the dataset line already exist, the new results will overwrite it.
        Structure :
        methods    | Metric1 | Metric2 | ...
        method1   |  0.85   |  12.4   | ...
        method2   |  0.82   |  11.9   | ...
        method3   |  0.88   |  12.9   | ...
        """
        save_path = self.path / f'{dataset}.csv'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metric_names = self.metrics.keys()
        metric_enriched_names = [m + (" \u2191" if metric.higher_is_better else " \u2193") for m, metric in self.metrics.items()]

        if not metric_names:
            print("No metrics to save.")
            return

        if self.count == 0:
            print("No samples counted.")
            return

        # Compute averages from aggregated sums
        averages = {name: self.results[name] / self.count[name] for name in metric_names}

        rows = []
        header = ["Methods"] + metric_enriched_names
        method_found = False

        # Load existing file if present
        if save_path.exists():
            with open(save_path, "r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if rows:
                header = rows[0]

            for i in range(1, len(rows)):
                if rows[i][0] == method:
                    rows[i] = [method] + [averages[m] for m in metric_names]
                    method_found = True
                    break

        # Append if dataset not found
        if not method_found:
            rows.append([method] + [averages[m] for m in metric_names])

        # Ensure header exists
        if not rows or rows[0] != header:
            rows.insert(0, header)

        # Write file
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)





