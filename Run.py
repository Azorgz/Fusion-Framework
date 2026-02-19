import gc
import os
import warnings

import torch
from Profile_Methods import ProfileMethods
from datasets import get_dataloaders
from methods import import_model
from options.options import Options


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    opt = Options().parse()
    if opt.task == 'profile':
        ProfileMethods(opt)
    else:
        methods = opt.name
        path_result = os.getcwd() + "/results/"
        dataLoaders = get_dataloaders(opt)
        print(f"Loaded datasets: {' | '.join(list(dataLoaders.keys()))}")
        for method in methods:
            model = import_model(method, opt, data=dataLoaders)
            model.run()
            gc.collect()
            torch.cuda.empty_cache()