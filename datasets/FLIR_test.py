from datasets.DatasetBase import DatasetBase


class FLIR_test(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "testC/"
        self.path_ir = self.root_dir + "testB/"

        super().__init__(opt)