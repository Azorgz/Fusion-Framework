import os

import numpy as np
from ImagesCameras import ImageTensor

from fusion_framework.datasets.DatasetBase import DatasetBase


def isimage(f: str):
    return f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))


class FLIR_selection(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"
    idx = [5556, 5570, 5650, 5690, 5985, 6081, 6165, 6244, 6273,
           7005, 7022, 7029, 7037, 7057, 7065, 7134, 7145, 7168,
           7206, 7208, 7213, 7227, 7306, 7370, 7394, 7408, 7432, 7435, 7523,
           7546, 7774, 7835, 7867, 7898, 7930, 8001, 8123, 8147, 8185, 8222, 8389, 8504, 8539, 8546,
           2346, 3500, 3567, 4335, 5217, 8794]

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC_0/"
        self.path_vis2 = self.root_dir + "trainA/"
        self.path_ir = self.root_dir + "trainB_0/"
        self.path_ir2 = self.root_dir + "trainA_T/"

        super().__init__(opt)
        list_vis = sorted([self.path_vis + '/' + f for f in os.listdir(self.path_vis) if isimage(f)] +
                          [self.path_vis2 + '/' + f for f in os.listdir(self.path_vis2) if isimage(f)])
        self.image_idx = list(range(len(list_vis)))
        list_vis = np.array(list_vis)[self.image_idx].tolist()
        self.image_vis = [f for idx, f in enumerate(list_vis) if (idx % opt.sampling == 0 and isimage(f))]
        list_ir = np.array(sorted([self.path_ir + '/' + f for f in os.listdir(self.path_ir) if isimage(f)] +
                           [self.path_ir2 + '/' + f for f in os.listdir(self.path_ir2) if isimage(f)]))[self.image_idx].tolist()
        self.image_ir = [f for idx, f in enumerate(list_ir) if (idx % opt.sampling == 0 and isimage(f))]
        assert len(self.image_vis) == len(self.image_ir), "Number of visible and infrared images must be equal."
        self.index = [i for i, f in enumerate(self.image_vis) if int(os.path.basename(f).split('_')[-1].split('.')[0]) in self.idx]

    def __getitem__(self, idx):
        idx_list = [i for i in self.index if i not in self.idx_ignore]
        idx = idx_list[idx % len(idx_list)]
        if self.direction == 'ir2vis':
            image_vis = ImageTensor(self.image_vis[idx])
            image_ir = ImageTensor(self.image_ir[idx]).RGB('gray').match_shape(image_vis)
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
        if 'tiff' in self.image_ir[idx].lower():
            image_ir = image_ir.normalize()
        if shape[0] > 1024 or shape[1] > 1024:
            shape_ = shape[0]//2, shape[1]//2
            if image_vis.shape[-2] != shape_[0] or image_vis.shape[-1] != shape_[1]:
                image_vis = image_vis.resize(shape_)
                image_ir = image_ir.resize(shape_)
            shape = shape_
        return image_vis, image_ir, shape
