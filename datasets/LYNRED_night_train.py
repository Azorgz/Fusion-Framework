from datasets.DatasetBase import DatasetBase


class LYNRED_NIGHT_TRAIN(DatasetBase):
    """
    Dataset class for the LYNRED day dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/LYNRED/LYNRED_datasets/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC/"
        self.path_ir = self.root_dir + "trainB/"

        super().__init__(opt)