from datasets.DatasetBase import DatasetBase


class EXP6(DatasetBase):
    """
    Dataset class for the FLIR night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq6/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)