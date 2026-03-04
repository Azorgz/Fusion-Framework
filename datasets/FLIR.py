from datasets.DatasetBase import DatasetBase


class FLIR(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC/"
        self.path_ir = self.root_dir + "trainB/"

        super().__init__(opt)
