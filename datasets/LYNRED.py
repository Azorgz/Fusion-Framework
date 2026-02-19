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


class LYNRED_DAY(DatasetBase):
    """
    Dataset class for the LYNRED day dataset.
    """
    root_dir = "/home/godeta/Images/ICCV/Data_publi/Lynred_day/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class LYNRED_NIGHT_TRAIN(DatasetBase):
    """
    Dataset class for the LYNRED day dataset.
    """
    root_dir = "/home/godeta/PycharmProjects/TIR2VIS/datasets/LYNRED/LYNRED_datasets/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "trainC/"
        self.path_ir = self.root_dir + "trainB/"

        super().__init__(opt)


class LYNRED_DAY_SAMPLES(DatasetBase):
    """
    Dataset class for the LYNRED day dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/LYNRED_DAY/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class LYNRED_NIGHT_SAMPLES(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/PycharmProjects/XCalib/examples/LYNRED_NIGHT/"

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"

        super().__init__(opt)


class LYNRED_EXP1(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq1/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)


class LYNRED_EXP2(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq2/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)


class LYNRED_EXP3(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq3/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)


class LYNRED_EXP4(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq4/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)


class LYNRED_EXP5(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq5/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)


class LYNRED_EXP6(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq6/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)


class LYNRED_EXP7(DatasetBase):
    """
    Dataset class for the LYNRED night dataset samples.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/seq7/"
    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        super().__init__(opt)