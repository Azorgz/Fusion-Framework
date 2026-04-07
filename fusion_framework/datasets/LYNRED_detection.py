import os

import yaml
from ImagesCameras import ImageTensor

from fusion_framework.datasets.DatasetBase import DatasetBase


class LYNRED_DETECTION(DatasetBase):
    """
    Dataset class for the LYNRED day dataset.
    """
    root_dir = "/home/godeta/Téléchargements/LYNRED_multimodal_detection_V1/detection_dataset/"
    idx = [2021, 2040, 2099, 2160, 2508, 2521, 2569, 2574, 2586, 2621, 2696, 2710, 2765, 2789, 2904,
           2911, 2974, 3091, 3127, 3176, 3232, 3521, 3523, 3528, 3538, 3603, 3619, 3732, 3761, 3803, 3852, 3913,
           2051, 2063, 2100, 2927, 2221, 2776, 2247, 2368, 2658, 2465]

    def __init__(self, opt):
        self.path_vis = self.root_dir + "visible/"
        self.path_ir = self.root_dir + "infrared_aligned/"
        # self.path_ir_16bits = self.root_dir + "infrared_16bits_aligned/"
        self.load_16bits = opt.load_16bits
        super().__init__(opt)
        self.index = [i for i, f in enumerate(self.image_vis) if
                      int(os.path.basename(f).split('_')[-1].split('.')[0]) in self.idx]
        self.epsilon = 1e-6

    @property
    def name(self):
        return f'{self.__class__.__name__}_{self.opt.sequence}'

    def __getitem__(self, idx):
        idx_list = [i for i in self.index if i not in self.idx_ignore]
        idx = idx_list[idx % len(idx_list)]
        if self.direction == 'ir2vis':
            image_vis = ImageTensor(self.image_vis[idx])
            image_ir = ImageTensor(self.image_ir[idx]).RGB('gray')
        else:
            image_ir = ImageTensor(self.image_ir[idx]).RGB('gray')
            image_vis = ImageTensor(self.image_vis[idx]).match_shape(image_ir)
        if self.crop != [0, 0, 0, 0]:
            image_ir = image_ir.crop(self.crop, mode='lrtb')
            image_vis = image_vis.crop(self.crop, mode='lrtb')
        if self.resize:
            image_vis = image_vis.resize(self.loadSize)
            shape = image_ir.shape[-2:]
            image_ir = image_ir.resize(self.loadSize)
        else:
            shape = image_ir.shape[-2:]
        image_vis[image_vis == 0] = self.epsilon
        return image_vis, image_ir, shape
