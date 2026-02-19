import os
import torch
import yaml
from datasets import get_dataloaders
from methods import benchmark_model


def ProfileMethods(opt):
    device = opt.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        device = "cpu"
    methods = opt.name
    opt.sampling = 1
    dataLoader = list(get_dataloaders(opt).values())[0]  # Use only the first dataset for profiling
    for method in methods:
        img_vis, img_ir, _ = dataLoader.__iter__().__next__()
        img_vis = img_vis.to(device)
        img_ir = img_ir.to(device)
        profiling_result = {f'{method}': benchmark_model(method, (img_vis, img_ir), device, opt)}
        print(f"Benchmarking results for {method}: {profiling_result}")
        file_name = os.getcwd() + "/methods/ProfilingResults.yaml"
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                profile = yaml.safe_load(f) or {}
        else:
            profile = {}
        profile.update(profiling_result)
        with open(file_name, "w") as f:
            yaml.dump(profile, f, default_flow_style=False)




