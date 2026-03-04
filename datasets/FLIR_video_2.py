from datasets.DatasetBase import DatasetBase


class FLIR_video_2(DatasetBase):
    """
    Dataset class for the FLIR video 2 dataset.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/video/clip2_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "/rgb/"
        if opt.load_16bits:
            self.path_ir = self.root_dir + "/ir_16bit/"
        else:
            self.path_ir = self.root_dir + "/ir_8bit/"
        super().__init__(opt)