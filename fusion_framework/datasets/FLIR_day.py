import os

from fusion_framework.datasets.DatasetBase import DatasetBase


class FLIR_DAY(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    # root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_day/"
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"
    idx = [1832]

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainA/"
        self.path_ir = self.root_dir + "trainA_T/"

        super().__init__(opt)
        self.index = [i for i, f in enumerate(self.image_vis) if int(os.path.basename(f).split('_')[-1].split('.')[0]) in self.idx]

    def __getitem__(self, index):
        img_vis, img_ir, shape = super().__getitem__(index)
        img_ir = (img_ir - img_ir.min()) / (img_ir.max() - img_ir.min())
        return img_vis, img_ir, shape