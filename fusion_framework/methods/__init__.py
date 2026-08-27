import importlib
import os
import shutil
import subprocess
from pathlib import Path
from ImagesCameras.tools.drawing import crop_to_common_roi as crop
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import yaml
from ImagesCameras import ImageTensor
from ImagesCameras.tools.misc import time_fct
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from fusion_framework.methods import Wrapping_and_Fusion
from fusion_framework.methods import Fusion, Wrapping
from fusion_framework.metrics import MetricModel

# region Models and Methods
MODELS = {'FUSION': Fusion.__methods__,
          'WRAPPING': Wrapping.__methods__,
          'WRAPPING_AND_FUSION': Wrapping_and_Fusion.__methods__}

FUSION_METHODS = MODELS['FUSION']
WRAPPING_METHODS = MODELS['WRAPPING']
WRAPPING_AND_FUSION_METHODS = MODELS['WRAPPING_AND_FUSION']
All_METHODS = FUSION_METHODS | WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS


def import_model(method, opt, task=None, data: dict[str: DataLoader] = None, path_result: str = None, **kwargs):
    """Import a model given its name. """
    with open(os.path.join(os.getcwd(), 'fusion_framework/methods', 'ProfilingResults.yaml'), 'r') as f:
        profile = yaml.safe_load(f)
    if isinstance(method, tuple):
        device = opt.device
        if device == "cuda:0" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            device = "cpu"
        wrapping, fusion = method
        wrapping_model = import_model(wrapping, opt, task='wrapping', **kwargs)
        fusion_model = import_model(fusion, opt, task='fusion', **kwargs)

        class dual_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.wrapping = wrapping_model.eval()
                self.fusion = fusion_model.eval()
                self.method = wrapping_model.method + '_' + fusion_model.method
                self.task = 'wrapping+fusion'
                self.crop = opt.crop_after if not opt.crop_common_roi else 'auto'
                self.resize = opt.resize_infer
                self.inferSize = opt.inferSize
                self.inverse_ir = hasattr(self.fusion, 'inverse_ir') and self.fusion.inverse_ir

            @torch.no_grad()
            def forward(self, img_vis, img_ir):
                if self.wrapping.model.direction == 'ir2vis':
                    img_vis, img_ir = self.wrapping(img_vis, img_ir)
                    img_ir = img_ir.match_shape(img_vis)
                elif self.wrapping.model.direction == 'vis2ir':
                    img_vis, img_ir = self.wrapping(img_vis, img_ir)
                    img_vis = img_vis.match_shape(img_ir)
                if self.crop == 'auto':
                    img_ir, img_vis = crop(img_ir, img_vis)
                else:
                    img_vis = img_vis.crop(self.crop, mode='lrtb') if self.crop is not None else img_vis
                    img_ir = img_ir.crop(self.crop, mode='lrtb') if self.crop is not None else img_ir
                if self.resize:
                    shape = img_vis.shape[-2:]
                    img_vis = img_vis.resize(self.inferSize)
                    img_ir = img_ir.resize(self.inferSize)
                fus, img_vis, img_ir = self.fusion(img_vis, img_ir)
                fus = ImageTensor(fus)
                if self.resize:
                    fus = fus.resize(shape)
                    img_vis = img_vis.resize(shape)
                    img_ir = img_ir.resize(shape)
                return fus, img_vis, img_ir

        return MasterModel(dual_model().eval(), data=data, opt=opt, device=device, path_result=path_result)
    else:
        mono = task is None
        device = opt.device
        if device == "cuda:0" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            device = "cpu"

        class mono_model(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.task = task
                self.model = get_model(device, opt, task=task, **kwargs).eval()
                self.method = method
                self.crop = opt.crop_after if not opt.crop_common_roi else 'auto'
                self.resize = opt.inferSize if opt.resize_infer else None
                if method in profile:
                    self.max_size = profile[method]['maximum_size']
                    self.exponential_fit = lambda y: profile[method]['exponential regression'][0]*y**profile[method]['exponential regression'][1]
                else:
                    self.max_size = None
                self.device = torch.device(device)
                self.inverse_ir = hasattr(self.model, 'inverse_ir') and self.model.inverse_ir

            @torch.no_grad()
            def forward(self, img_vis, img_ir):
                h, w = img_vis.shape[-2:]
                if self.max_size is not None:
                    memory_available = get_free_gpu_memory()[self.device.index or 0] * 0.9  # Use only 90% of available memory
                    estimated_max_size = int(self.exponential_fit(memory_available))
                    max_size = min(self.max_size, estimated_max_size)
                    if h * w > max_size**2:
                        size = (int(h * (max_size / max(h, w))),
                                int(w * (max_size / max(h, w))))
                        if self.task == 'wrapping':
                            img_vis, img_ir = self.model(img_vis.resize(size), img_ir.resize(size))
                            img_vis, img_ir = img_vis.resize((h, w)), img_ir.resize((h, w))
                            if self.crop == 'auto':
                                img_ir, img_vis = crop(img_ir, img_vis)
                            else:
                                img_vis = img_vis.crop(self.crop, mode='lrtb') if self.crop is not None else img_vis
                                img_ir = img_ir.crop(self.crop, mode='lrtb') if self.crop is not None else img_ir
                            return img_vis, img_ir
                        else:
                            out, img_vis, img_ir = self.model(img_vis.resize(size), img_ir.resize(size))
                            out = ImageTensor(out).resize((h, w))
                            img_vis = img_vis.resize((h, w))
                            img_ir = img_ir.resize((h, w))
                            return out, img_vis, img_ir
                    else:
                        if self.task == 'wrapping':
                            img_vis, img_ir = self.model(img_vis, img_ir)
                            if self.crop == 'auto':
                                img_ir, img_vis = crop(img_ir, img_vis)
                            else:
                                img_vis = img_vis.crop(self.crop, mode='lrtb') if self.crop is not None else img_vis
                                img_ir = img_ir.crop(self.crop, mode='lrtb') if self.crop is not None else img_ir
                            return img_vis, img_ir
                        else:
                            out, img_vis, img_ir = self.model(img_vis, img_ir)
                            out = ImageTensor(out)
                            return out, img_vis, img_ir
                else:
                    if self.task == 'wrapping':
                        img_vis, img_ir = self.model(img_vis, img_ir)
                        if self.crop == 'auto':
                            img_ir, img_vis = crop(img_ir, img_vis)
                        else:
                            img_vis = img_vis.crop(self.crop, mode='lrtb') if self.crop is not None else img_vis
                            img_ir = img_ir.crop(self.crop, mode='lrtb') if self.crop is not None else img_ir
                        return img_vis, img_ir
                    else:
                        if self.resize is not None:
                            img_vis = img_vis.resize(self.resize)
                            img_ir = img_ir.resize(self.resize)
                        out, img_vis, img_ir = self.model(img_vis, img_ir)
                        out = ImageTensor(out)
                        return out, img_vis, img_ir

            @torch.no_grad()
            def forward_wo_resize(self, img_vis, img_ir):
                out, img_vis, img_ir = self.model(img_vis, img_ir)
                out = ImageTensor(out)
                return out, img_vis, img_ir

        # PURE FUSION METHODS
        if method in FUSION_METHODS:
            module = importlib.import_module(f"{FUSION_METHODS[method]}", package=__package__)
            method = FUSION_METHODS[method].split('.')[-1]
            get_model = module.get_model
            task = 'fusion' if task is None else task

        # PURE WRAPPING METHODS
        elif method in WRAPPING_METHODS:
            module = importlib.import_module(f"{WRAPPING_METHODS[method]}", package=__package__)
            method = WRAPPING_METHODS[method].split('.')[-1]
            get_model = module.get_model
            task = 'wrapping' if task is None else task

        # WRAPPING + FUSION METHODS
        elif method in WRAPPING_AND_FUSION_METHODS:
            module = importlib.import_module(f"{WRAPPING_AND_FUSION_METHODS[method]}", package=__package__)
            method = WRAPPING_AND_FUSION_METHODS[method].split('.')[-1]
            get_model = module.get_model
            task = opt.task

        else:
            raise ValueError(f"Method {method} not recognized. Please choose a method in {All_METHODS.keys()}")
        if mono:
            return MasterModel(mono_model().eval(), data=data, opt=opt, device=device, path_result=path_result)
        return mono_model().eval()


class MasterModel(nn.Module):
    def __init__(self, model: nn.Module = None, path_result: str = None,
                 data: dict[str: DataLoader] = None,
                 opt=None, device=None):
        super().__init__()
        self.task = model.task
        self.model = model
        self.metrics = MetricModel(opt.metrics, device)
        self.train(model.training)
        self.method = model.method
        self.path_result = path_result + self.method
        self.data = data
        self.opt = opt
        self.device = device
        self.resize = opt.resize_save

    def run(self):
        for dataset_name, dataloader in self.data.items():
            path = self.path_result + "/" + dataset_name + "/"
            if self.opt.reset_result:
                shutil.rmtree(path, ignore_errors=True)
                res_list = []
            else:
                res_list = [f.split('.')[0] for f in os.listdir(path)] if os.path.exists(path) else []
            os.makedirs(path, exist_ok=True)
            if self.task == 'wrapping':
                os.makedirs(path + '/ir', exist_ok=True)
                os.makedirs(path + '/vis', exist_ok=True)
            dataloader.dataset.update(res_list)
            if 'XCalib' in self.method:
                if hasattr(self.model, 'model'):
                    self.model.model.fit_to_data(dataloader)
                else:
                    self.model.wrapping.model.fit_to_data(dataloader)
            self.metrics._init_results()
            for *imgs, ori_size in tqdm(dataloader, desc=f"Running {self.method.replace('_', ' ')} on dataset {dataset_name}"):
                if len(imgs) == 3:
                    img_vis, img_ir, other = imgs
                else:
                    img_vis, img_ir = imgs
                if img_vis is not None:
                    name_img = img_vis.name
                    img_vis = img_vis.to(self.device)
                    img_ir = img_ir.to(self.device)
                    with torch.no_grad():
                        if self.task == 'wrapping':
                            img_vis, img_ir = self(img_vis, img_ir)
                            if self.resize:
                                img_vis = ImageTensor(img_vis).resize(ori_size[0])
                                img_ir = ImageTensor(img_ir).resize(ori_size[0])
                            # ImageTensor(img_vis).save(path + '/vis', name=name_img, ext="png", depth=8)
                            ImageTensor(img_ir).GRAY().save(path + '/ir', name=name_img, ext="png", depth=8)
                            # self.metrics(vis=img_vis, ir=img_ir, fus=None)
                        else:
                            fus, vis, ir = self(img_vis, img_ir)
                            if self.resize:
                                fus = ImageTensor(fus).resize(ori_size[0])
                            else:
                                fus = ImageTensor(fus)
                            fus.save(path, name=name_img, ext="png", depth=8)
                            if self.model.inverse_ir:
                                ir = 1 - ir
                            self.metrics(vis, ir, fus)
            self.metrics.save_results(method=self.method, dataset=dataset_name)

    @torch.no_grad()
    def forward(self, img_vis, img_ir):
        return self.model(img_vis.to(self.device), img_ir.to(self.device))

    @torch.no_grad()
    def forward_wo_resize(self, img_vis, img_ir):
        return self.model.forward_wo_resize(img_vis, img_ir)
# endregion


# region Benchmarking
def benchmark_model(method, data, device, opt, **kwargs):
    model = import_model(method, opt, data=data, **kwargs)
    img_vis, img_ir = data
    print(f"Benchmarking {model.method} for increasing image sizes until OOM...")

    def bench_data(model, img_vis, img_ir, rep):
        size = (128, 128)
        time = {}
        memory = {}
        cond = 1
        maximum_size = 0
        try:
            while cond:
                print(f"Testing size: {size[0]}x{size[1]}...")
                if hasattr(model, 'fixed_size'):
                    cond = 0
                    size = model.fixed_size
                    maximum_size = size[0]
                img_vis_resized = img_vis.resize(size).to(device)
                img_ir_resized = img_ir.resize(size).to(device)
                torch.cuda.reset_peak_memory_stats()
                _, time_rep = (time_fct(model.forward_wo_resize, reps=rep, exclude_first=True, verbose=False)
                               (img_vis_resized, img_ir_resized))
                time[f'{size[0]}px'] = f'{time_rep} sec / it'
                # Get peak memory
                peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
                memory[f'{size[0]}px'] = f'{peak_mem} MiB'
                size = (size[0] * 2, size[1] * 2)
        except torch.OutOfMemoryError:
            maximum_size = size[0] // 2
        return {'Process time': time, 'Memory Usage': memory, 'maximum_size': maximum_size}

    bench_data_dict = bench_data(model, img_vis, img_ir, 5)
    bench_data_dict['exponential regression'] = fit_powers(bench_data_dict['Memory Usage'])
    return bench_data_dict


def get_free_gpu_memory():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits']
    ).decode('utf-8')
    memory_info = []
    for line in result.strip().split('\n'):
        total, used = map(int, line.split(','))
        free = total - used
        memory_info.append(free)
    return memory_info


def fit_exponentials(memory_data):
    data = [(int(size.replace('px', '')), float(memory.replace(' MiB', ''))) for size, memory in memory_data.items()]
    data.sort(key=lambda x: x[0])
    x = np.array([size for size, memory in data]).reshape(-1, 1)
    y = np.array([memory for size, memory in data]).reshape(-1, 1)
    # Transformation logarithmique
    lnx = np.log(x)

    # Régression linéaire
    model = LinearRegression()
    model.fit(y, lnx)

    b = float(model.coef_[0])
    ln_a = model.intercept_
    a = float(np.exp(ln_a))
    return [a, b]

def fit_powers(memory_data):
    data = [(int(size.replace('px', '')), float(memory.replace(' MiB', ''))) for size, memory in memory_data.items()]
    data.sort(key=lambda x: x[0])
    x = np.array([size for size, memory in data]).reshape(-1, 1)
    y = np.array([memory for size, memory in data]).reshape(-1, 1)
    # Transformation log-log
    logx = np.log(x).reshape(-1, 1)
    logy = np.log(y)

    # Régression linéaire
    model = LinearRegression()
    model.fit(logy, logx)

    b = float(model.coef_.squeeze())
    ln_a = model.intercept_
    a = float(np.exp(ln_a).squeeze())
    return [a, b]


def fit_poly(memory_data, degree=2):
    data = [(int(size.replace('px', '')), float(memory.replace(' MiB', ''))) for size, memory in memory_data.items()]
    data.sort(key=lambda x: x[0])
    x = np.array([size for size, memory in data])
    y = np.array([memory for size, memory in data])

    # Ajustement polynomial
    coeffs = np.polyfit(y, x, degree)
    return [float(c) for c in coeffs]  # Retourne les coefficients du polynôme

# endregion