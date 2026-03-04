from datasets.DatasetBase import DatasetBase


class MSRS(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/MSRS-main/test/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vi/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)