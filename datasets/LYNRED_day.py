from datasets.DatasetBase import DatasetBase


class LYNRED_DAY(DatasetBase):
    """
    Dataset class for the LYNRED day dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/Lynred_day/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)
