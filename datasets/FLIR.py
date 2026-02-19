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


class FLIR(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/FLIR/FLIR_datasets/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC/"
        self.path_ir = self.root_dir + "trainB/"

        super().__init__(opt)


class FLIR_DAY(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_day/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class FLIR_NIGHT(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class FLIR_reg_day(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_day/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir_reg_ours/"

        super().__init__(opt)


class FLIR_reg_night(DatasetBase):
    """
    Dataset class for the FLIR dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/FLIR_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir_reg_ours/"

        super().__init__(opt)


class FLIR_DAY_SAMPLES(DatasetBase):
    """
    Dataset class for the FLIR day dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/FLIR_DAY/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class FLIR_NIGHT_SAMPLES(DatasetBase):
    """
    Dataset class for the FLIR night dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/FLIR_NIGHT/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class FLIR_video_1(DatasetBase):
    """
    Dataset class for the FLIR video 1 dataset.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/video/clip1_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "/rgb/"
        if opt.load_16bits:
            self.path_ir = self.root_dir + "/ir_16bit/"
        else:
            self.path_ir = self.root_dir + "/ir_8bit/"
        super().__init__(opt)


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


class FLIR_video_3(DatasetBase):
    """
    Dataset class for the FLIR video 3 dataset.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/video/clip3_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "/rgb/"
        if opt.load_16bits:
            self.path_ir = self.root_dir + "/ir_16bit/"
        else:
            self.path_ir = self.root_dir + "/ir_8bit/"
        super().__init__(opt)


class FLIR_video_4(DatasetBase):
    """
    Dataset class for the FLIR video 4 dataset.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/FLIR_ADAS_1_3_full/video/clip4_night/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "/rgb/"
        if opt.load_16bits:
            self.path_ir = self.root_dir + "/ir_16bit/"
        else:
            self.path_ir = self.root_dir + "/ir_8bit/"
        super().__init__(opt)


class Extract_FLIR(DatasetBase):
    """
    Dataset class for extracting images from the FLIR dataset.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/from_FLIR/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis"
        self.path_ir = self.root_dir + "ir"

        super().__init__(opt)


class ROADSCENE(DatasetBase):
    """
    Dataset class for the ROADSCENE dataset.
    """
    root_dir = "/media/godeta/T5 EVO/Datasets/RoadScene-master/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "crop_HR_visible/"
        self.path_ir = self.root_dir + "cropinfrared/"

        super().__init__(opt)