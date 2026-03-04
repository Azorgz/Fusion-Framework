from datasets.DatasetBase import DatasetBase


class LYNRED_NIGHT(DatasetBase):
    """
    Dataset class for the LYNRED night dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/Lynred_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir_reg_ours/"

        super().__init__(opt)