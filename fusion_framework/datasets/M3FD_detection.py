import os

from fusion_framework.datasets.DatasetBase import DatasetBase


class M3FD_detection(DatasetBase):
    """
    Dataset class for M3FD, detection part.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/M3FD/Detection/"
    idx = [0, 117, 149, 274, 829, 334, 976, 1115, 1165, 1492, 2531, 2663, 3880, 4053]

    def __init__(self, opt):
        self.path_vis = self.root_dir + "Vis/"
        self.path_ir = self.root_dir + "Ir/"
        self.annotation_file = self.root_dir + "Annotations/"
        super().__init__(opt)
        self.index = [i for i, f in enumerate(self.image_vis) if
                      int(os.path.basename(f).split('_')[-1].split('.')[0]) in self.idx]