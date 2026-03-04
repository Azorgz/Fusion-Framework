from datasets.DatasetBase import DatasetBase


class FLIR_selection(DatasetBase):
    """
    Dataset class for extracting images from the FLIR dataset.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/from_FLIR/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis"
        self.path_ir = self.root_dir + "ir"

        super().__init__(opt)