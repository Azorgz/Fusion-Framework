import os

import torch
from ImagesCameras import ImageTensor
from ImagesCameras.Metrics import METRICS_DICT
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

from tqdm import tqdm

from fusion_framework.datasets.TestLightness import TestLightness

path = "/home/godeta/Bureau/selection sequence/test_lightness/"
ref_list_vis = [path + 'vis/' + f for f in sorted(os.listdir(path + 'vis/'))]
ref_list_ir = [path + 'ir/' + f for f in sorted(os.listdir(path + 'ir/'))]
list_noisy = [path + 'noisy/' + f for f in sorted(os.listdir(path + 'noisy/'))]

path_results = "/home/godeta/PycharmProjects/FusionMethods/results/"
methods = ['Alpha_blending', 'NightToDay', 'SeAFusion', 'TarDAL', 'TextIF']  # ,'MaeFuse','SAGE', 'PAIF'

metric_names = ['psnr', 'mse', 'ssim', 'ms_ssim', 'gc', 'nec', 'vif', 'mi']
n_images = len(ref_list_vis)
n_noise = TestLightness.night_levels  # number of noise levels in the TestLightness dataset

results = defaultdict(lambda: defaultdict(dict))

compute_metrics = False  # Set to False if you only want to recompute the relative values and save the CSVs again
draw_results = True  # Set to True if you want to draw the results (e.g., using matplotlib or seaborn)


if __name__ == "__main__":

    if compute_metrics:
        def compute_relative_row(base_row, out_row):
            rel = {}
            for m in base_row.keys():
                base = base_row[m]
                out = out_row[m]

                if METRICS_DICT[m].higher_is_better:
                    rel[m] = (out - base) / (abs(base) + 1e-8)
                else:
                    rel[m] = (base - out) / (abs(base) + 1e-8)
            return rel


        metrics = {m: METRICS_DICT[m](device=torch.device('cuda')) for m in metric_names}
        bar = tqdm(total=len(methods) * n_images * n_noise * len(metrics), desc="Computing metrics", unit="metric")
        for method in methods:

            rows = []
            # store relative values for averaging
            rel_accumulator = {noise: [] for noise in range(n_noise)}
            bar.set_description(f"Processing {method}")
            list_output = [path_results + method + '/TestLightness/' + f for f in sorted(os.listdir(path_results + method + '/TestLightness/'))]
            for img_id in range(n_images):

                input_scores = []
                output_scores = []

                for noise_level in range(n_noise):

                    clean = ImageTensor(ref_list_vis[img_id]).to('cuda')
                    noisy = ImageTensor(list_noisy[img_id * n_noise + noise_level]).to('cuda')
                    output = ImageTensor(list_output[img_id * n_noise + noise_level]).to('cuda')

                    metrics_input = {}
                    metrics_output = {}

                    for name, metric in metrics.items():
                        val_input = metric(clean, noisy, mask=clean > 0)
                        val_output = metric(clean, output, mask=clean > 0)

                        metrics_input[name] = val_input.cpu().numpy()
                        metrics_output[name] = val_output.cpu().numpy()
                        bar.update(1)

                    input_scores.append(metrics_input)
                    output_scores.append(metrics_output)

                    # ---- INPUT row ----
                    row_in = {
                        "image": img_id,
                        "type": "input",
                        "noise": noise_level,
                        **metrics_input
                    }
                    rows.append(row_in)

                    # ---- OUTPUT row ----
                    row_out = {
                        "image": img_id,
                        "type": "output",
                        "noise": noise_level,
                        **metrics_output
                    }
                    rows.append(row_out)

                    # ---- RELATIVE ----
                    rel_row = compute_relative_row(metrics_input, metrics_output)
                    rel_accumulator[noise_level].append(rel_row)

                results[method][img_id]["input"] = input_scores
                results[method][img_id]["output"] = output_scores

            # ---- AVERAGE RELATIVE ROWS ----
            for noise_level in range(n_noise):
                rel_list = rel_accumulator[noise_level]

                avg_rel = {}
                for m in metric_names:
                    avg_rel[m] = np.mean([r[m] for r in rel_list])

                row_avg = {
                    "image": "average",
                    "type": "relative",
                    "noise": noise_level,
                    **avg_rel
                }
                rows.append(row_avg)

            # ---- SAVE CSV ----
            df = pd.DataFrame(rows)

            # Optional: enforce column order
            df = df[["image", "type", "noise"] + metric_names]

            df.to_csv(f"{path_results}/Metrics/night_test/{method}.csv", index=False)

    if draw_results:
        import os
        import glob
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import FuncFormatter


        def percent_tick(x, pos=0):
            return f"{x*10:1.0f}%"

        def find_grid(n):
            """Trouve une grille adaptée pour n sous-graphiques"""
            if n <= 3:
                return 1, n
            b = int(np.ceil(np.sqrt(n)))
            a = int(np.ceil(n / b))
            return a, b

        path = "/home/godeta/PycharmProjects/FusionMethods/results/Metrics/night_test/"

        # 1. Configuration des métriques et des fichiers
        csv_files = sorted(glob.glob(path + "*.csv"))
        csv_files.reverse()
        metrics = ['psnr', 'ms_ssim', 'mi']
        metrics_full_names = ['Peak Signal-to-Noise Ratio (PSNR)', 'Multi-Scale Structural Similarity (MS-SSIM)', 'Mutual Information (MI)']
        # metrics_full_names = ['Peak Signal-to-Noise Ratio (PSNR)', 'Mean Squared Error (MSE)',
        #                       'Multi-Scale Structural Similarity (MS-SSIM)', 'Normalized Edges Correlation (NEC)', 'Mutual Information (MI)', 'Visual Information Fidelity (VIF)']
        images_list = [str(i) for i in range(9)]  # Filtre pour inclure uniquement les images de '0' à '8'

        methods_data = {}
        reference_data = {}
        methods_names = []

        # 2. Chargement et agrégation des données
        for file in csv_files:
            method_name = file.replace(".csv", "").split("/")[-1]
            df = pd.read_csv(file)

            # On filtre pour exclure la ligne 'average' et ne garder que le cœur des données
            df_numeric = df[df['image'].isin(images_list)]

            # Calcul de la moyenne des 'output' pour chaque méthode par niveau de bruit
            df_output = df_numeric[df_numeric['type'] == 'output']
            output_means = df_output.groupby('noise')[metrics].mean()

            for metric in metrics:
                if metric not in methods_data:
                    methods_data[metric] = {}
                methods_data[metric][method_name] = output_means[metric]

            # Extraction de la référence 'input' (identique d'un fichier à l'autre)
            df_input = df_numeric[df_numeric['type'] == 'input']
            input_means = df_input.groupby('noise')[metrics].mean()
            for metric in metrics:
                reference_data[metric] = input_means[metric]

        # 3. Génération du graphique combiné (Grille 2x3)
        sns.set_theme(style="whitegrid")
        grid = find_grid(len(metrics))
        fig, axes = plt.subplots(*grid, figsize=(grid[1]*6, grid[0]*5 + 1))
        axes = axes.flatten()

        # Liste de marqueurs pour différencier facilement les courbes
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '<']

        for i, (metric, name) in enumerate(zip(metrics, metrics_full_names)):
            ax = axes[i]

            # Tracé de la référence Input
            ax.plot(reference_data[metric].index, reference_data[metric].values,
                    label='Input (Reference)', color='black', linestyle='--', linewidth=2.5, marker='o')

            # Tracé des courbes de chaque méthode fusionnée
            for j, (method_name, series) in enumerate(methods_data[metric].items()):
                m = markers[(j + 1) % len(markers)]
                ax.plot(series.index, series.values, label=method_name, linewidth=1.5, marker=m)

            ax.set_title(f"{name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Alpha", fontsize=10)
            ax.xaxis.set_major_formatter(FuncFormatter(percent_tick))
            # if i == len(metrics) - 1:  # Seule la dernière case a la légende
            ax.legend(fontsize=10)

        # Nettoyage de la 6ème case vide de la grille
        index_unused = [i for i in range(len(axes)) if i >= len(metrics)]
        for idx in index_unused:
            fig.delaxes(axes[idx])
        plt.tight_layout()
        plt.savefig("metrics.png", dpi=300)
        plt.close()

        # # 4. Génération et sauvegarde des graphiques individuels séparés
        # for metric in metrics:
        #     plt.figure(figsize=(9, 5.5))
        #
        #     # Référence
        #     plt.plot(reference_data[metric].index, reference_data[metric].values,
        #              label='Input (Référence)', color='black', linestyle='--', linewidth=2.5, marker='o')
        #
        #     # Méthodes
        #     for j, (method_name, series) in enumerate(methods_data[metric].items()):
        #         m = markers[(j + 1) % len(markers)]
        #         plt.plot(series.index, series.values, label=method_name, marker=m, alpha=0.85)
        #
        #     plt.title(f"Évolution de {metric.upper()} en fonction du bruit", fontsize=13, fontweight='bold')
        #     plt.xlabel("Niveau de bruit")
        #     plt.ylabel(metric.upper())
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende déportée pour plus de clarté
        #     plt.tight_layout()
        #     plt.savefig(f"metric_{metric}.png", dpi=300)
        #     plt.close()

        print("Le graphique a été généré avec succès !")