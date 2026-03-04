from datasets.DatasetBase import DatasetBase


class ROADSCENE(DatasetBase):
    """
    Dataset class for the ROADSCENE dataset.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/RoadScene-master/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "crop_HR_visible/"
        self.path_ir = self.root_dir + "cropinfrared/"

        super().__init__(opt)