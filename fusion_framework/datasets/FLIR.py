import os

from fusion_framework.datasets.DatasetBase import DatasetBase


class FLIR(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    # root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"
    root_dir = "/media/godeta/T5 EVO/Datasets/FLIR/FLIR_Aligned/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "images_rgb_train/data"
        # self.path_vis = self.root_dir + "trainC/"
        # self.path_ir = self.root_dir + "trainB/"
        self.path_ir = self.root_dir + "images_thermal_train/data"
        super().__init__(opt)
