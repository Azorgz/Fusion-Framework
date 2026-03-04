from datasets.DatasetBase import DatasetBase


class FLIR_NIGHT_SAMPLES(DatasetBase):
    """
    Dataset class for the FLIR night dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/FLIR_NIGHT/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)