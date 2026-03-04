from datasets.DatasetBase import DatasetBase


class FLIR_DAY(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_day/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)