from fusion_framework.datasets.DatasetBase import DatasetBase


class M3FD_fusion(DatasetBase):
    """
    Dataset class for M3FD, fusion part.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/M3FD/Fusion/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "Vis/"
        self.path_ir = self.root_dir + "Ir/"
        self.annotation_file = self.root_dir + "Annotations/"
        super().__init__(opt)