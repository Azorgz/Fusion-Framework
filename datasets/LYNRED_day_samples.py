from datasets.DatasetBase import DatasetBase


class LYNRED_DAY_SAMPLES(DatasetBase):
    """
    Dataset class for the LYNRED day dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/LYNRED_DAY/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)