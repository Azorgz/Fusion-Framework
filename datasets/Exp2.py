from datasets.DatasetBase import DatasetBase


class EXP2(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq2/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)