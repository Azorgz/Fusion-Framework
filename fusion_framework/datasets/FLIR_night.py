import os

from ImagesCameras import ImageTensor

from fusion_framework.datasets.DatasetBase import DatasetBase


class FLIR_NIGHT(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"
    # root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC/"
        self.path_ir = self.root_dir + "trainB/"
        self.path_mask = self.root_dir + "FLIR_IR_seg_mask/"

        super().__init__(opt)
        self.image_mask = [os.path.join(self.path_mask, f) for f in sorted(os.listdir(self.path_mask))]
        self.index = [i for i in range(len(self.image_vis))]
        # read the text file to find the index of the image to load
        with open(os.path.join(self.root_dir, "traffic_light_colors.txt"), 'r') as f:
            idx = [int(line.strip()) for line in f.readlines()]
        self.idx_list = [i for i in self.index if i in idx]

    def __getitem__(self, idx):
        idx = self.idx_list[idx % len(self.idx_list)]
        image_ir = ImageTensor(self.image_ir[idx]).RGB('gray')
        image_vis = ImageTensor(self.image_vis[idx]).match_shape(image_ir)
        # image_mask = ImageTensor(self.image_mask[idx]).GRAY()
        crop_ratio_w = 1 / ((500 - 360) / 2 / 500)
        crop_ratio_h = 1 / ((400 - 288) / 2 / 400)
        x = int(image_ir.shape[3] // crop_ratio_w)
        y = int(image_ir.shape[2] // crop_ratio_h)
        image_ir = image_ir.crop((x, x, y, y), mode='lrtb')
        image_vis = image_vis.crop((x, x, y, y), mode='lrtb')
        shape_ratio = image_ir.shape[-2] / image_ir.shape[-1]
        if shape_ratio != 288 / 360:
            if shape_ratio > 288 / 360:
                new_w = int(image_ir.shape[-2] / (288 / 360))
                image_ir = image_ir.resize((image_ir.shape[-2], new_w))
                image_vis = image_vis.resize((image_vis.shape[-2], new_w))
            else:
                new_h = int(image_ir.shape[-1] * (288 / 360))
                image_ir = image_ir.resize((new_h, image_ir.shape[-1]))
                image_vis = image_vis.resize((new_h, image_vis.shape[-1]))
        # image_mask = image_mask.match_shape(image_ir)
        return image_vis, image_ir, image_ir.shape[-2:]
