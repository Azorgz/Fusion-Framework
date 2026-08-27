import os

import numpy as np
from torch.utils.data import Dataset
from ImagesCameras import ImageTensor


def isimage(f: str):
    return f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))


class DatasetBase(Dataset):
    """
    Base class for datasets.
    """
    path_vis = ""
    path_ir = ""
    is_aligned = True
    first_image_idx = 0
    last_image_idx = -1
    image_idx = None
    crop = [0, 0, 0, 0]  # [left, right, top, bottom]

    def __init__(self, opt):
        self.opt = opt
        self.loadSize = opt.loadSize
        self.resize = opt.resize_load
        self.direction = opt.direction
        self.idx_ignore = []
        opt.sampling = opt.sampling if opt.sampling > 0 else 1
        list_vis = [f for f in sorted(os.listdir(self.path_vis)) if isimage(f)]
        if self.image_idx is None:
            if self.last_image_idx == -1:
                self.last_image_idx = len(list_vis)
            self.image_idx = [i for i in range(self.first_image_idx, self.last_image_idx)]
        list_vis = np.array(list_vis)[self.image_idx].tolist()
        self.image_vis = [os.path.join(self.path_vis, f) for idx, f in enumerate(list_vis) if
                          (idx % opt.sampling == 0 and isimage(f))]
        list_ir = np.array([f for f in sorted(os.listdir(self.path_ir)) if isimage(f)])[self.image_idx].tolist()
        self.image_ir = [os.path.join(self.path_ir, f) for idx, f in enumerate(list_ir) if
                         (idx % opt.sampling == 0 and isimage(f))]
        if hasattr(self, 'path_ir_16bits'):
            list_ir_16bits = np.array([f for f in sorted(os.listdir(self.path_ir)) if isimage(f)])[self.image_idx].tolist()
            self.image_ir_16bits = [os.path.join(self.path_ir_16bits, f) for idx, f in enumerate(list_ir_16bits) if
                                    (idx % opt.sampling == 0 and isimage(f))]
        assert len(self.image_vis) == len(self.image_ir), "Number of visible and infrared images must be equal."
        self.crop = opt.crop_before if opt.crop_before else self.crop  # [left, right, top, bottom]
        self.index = [i for i in range(len(self.image_vis))]
        self.idx_list = [i for i in self.index if i not in self.idx_ignore]

    @property
    def name(self):
        return str(self.__class__.__name__)

    def __len__(self):
        return len(self.index) - len(self.idx_ignore)

    def __getitem__(self, idx):
        idx = self.idx_list[idx % len(self.idx_list)]
        if self.direction == 'ir2vis':
            image_vis = ImageTensor(self.image_vis[idx])
            image_ir = ImageTensor(self.image_ir[idx]).RGB('gray').match_shape(image_vis)
        else:
            image_ir = ImageTensor(self.image_ir[idx]).RGB('gray')
            image_vis = ImageTensor(self.image_vis[idx]).match_shape(image_ir)
        if self.crop != [0, 0, 0, 0]:
            image_ir = image_ir.crop(self.crop, mode='lrtb')
            image_vis = image_vis.crop(self.crop, mode='lrtb')
        shape = image_ir.shape[-2:]
        if self.resize:
            image_vis = image_vis.resize(self.loadSize)
            shape = image_ir.shape[-2:]
            image_ir = image_ir.resize(self.loadSize)
        return image_vis, image_ir, shape  # Return original size for resizing later

    def update(self, list_ignore):
        self.idx_ignore = [i for i, img in enumerate(self.image_vis) if img.split('/')[-1].split('.')[0] in list_ignore]
