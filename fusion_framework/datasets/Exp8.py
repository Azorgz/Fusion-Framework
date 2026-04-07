from fusion_framework.datasets.DatasetBase import DatasetBase


class EXP8(DatasetBase):
    """
    Dataset class for the FLIR night dataset samples.
    """
    root_dir = "//home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC/"
        self.path_ir = self.root_dir + "trainB/"
        self.first_image_idx = 468
        self.last_image_idx = 706
        super().__init__(opt)