import os
from torch.utils.data import Dataset
from ImagesCameras import ImageTensor


class DatasetBase(Dataset):
    """
    Base class for datasets.
    """
    path_vis = ""
    path_ir = ""

    def __init__(self, opt):
        self.loadSize = opt.loadSize
        self.resize = opt.resize
        self.direction = opt.direction
        self.idx_ignore = []
        opt.sampling = opt.sampling if opt.sampling > 0 else 1
        self.image_vis = [os.path.join(self.path_vis, f) for idx, f
                          in enumerate(sorted(os.listdir(self.path_vis))) if idx % opt.sampling == 0]
        self.image_ir = [os.path.join(self.path_ir, f) for idx, f
                         in enumerate(sorted(os.listdir(self.path_ir))) if idx % opt.sampling == 0]
        assert len(self.image_vis) == len(self.image_ir), "Number of visible and infrared images must be equal."
        self.crop = opt.crop

    def __len__(self):
        return len(self.image_vis) - len(self.idx_ignore)

    def __getitem__(self, idx):
        if idx in self.idx_ignore:
            return self.__getitem__((idx + 1) % len(self))
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
        return image_vis, image_ir, shape  # Return original size for resizing later

    def update(self, list_ignore):
        self.idx_ignore = [i for i, img in enumerate(self.image_vis) if img.split('/')[-1].split('.')[0] in list_ignore]
