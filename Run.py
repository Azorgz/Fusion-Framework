import gc
import os
import warnings

import torch
from Profile_Methods import ProfileMethods
from fusion_framework.datasets import get_dataloaders
from fusion_framework.methods import import_model
from fusion_framework.options.options import Options

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    opt = Options().parse()
    if opt.task == 'profile':
        ProfileMethods(opt)
    else:
        methods = opt.name
        path_result = ROOT_DIR + "/results/"
        dataLoaders = get_dataloaders(opt)
        print(f"Loaded datasets: {' | '.join(list(dataLoaders.keys()))}")
        for method in methods:
            model = import_model(method, opt, data=dataLoaders, path_result=path_result)
            model.run()
            gc.collect()
            torch.cuda.empty_cache()
