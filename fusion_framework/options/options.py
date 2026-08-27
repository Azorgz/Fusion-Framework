import argparse
import os
from typing import Tuple, List

from ImagesCameras.Metrics import METRICS_DICT

from fusion_framework.datasets import DATASETS
from fusion_framework.methods import All_METHODS, FUSION_METHODS, WRAPPING_AND_FUSION_METHODS, WRAPPING_METHODS


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # Method parameters
        # 'Alpha_blending', 'SeAFusion', 'TarDAL', 'TextIF', 'MaeFuse', 'SAGE', 'PAIF', 'NightToday'
        self.parser.add_argument('--name', default=['GIFNet'], type=str | list | tuple, choices=All_METHODS, help='name of the method tested')
        self.parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'], help='cuda or cpu according your material')
        self.parser.add_argument('--task', default='fusion', type=str, choices=['auto', 'fusion', 'wrapping', 'wrapping_and_fusion', 'profile'], help='fusion ,wrapping or wrapping&fusion')
        self.parser.add_argument('--direction', default='vis2ir', type=str, choices=['vis2ir', 'ir2vis'], help='vis2ir or ir2vis')

        # Datasets - DataLoader
        self.parser.add_argument('--dataset', default=['FLIR_reg_night'], type=str or list, choices=list(DATASETS.keys()), help='Name of the dataset used for testing')
        self.parser.add_argument('--sequence', default=0, type=int, help='only for wrapping')
        self.parser.add_argument('--reset_result', default=True, type=bool, help='Erase the previous results for the given dataset/method')
        self.parser.add_argument('--shuffle', default=False, type=bool, help='Shuffle the input dataset')
        self.parser.add_argument('--sampling', default=25, type=int, help='Dataset subsampling')
        self.parser.add_argument('--resize_load', default=True, type=bool, help='Need to resize the loaded image')
        self.parser.add_argument('--resize_infer', default=True, type=bool, help='Need to resize the input for fusion')
        self.parser.add_argument('--resize_save', default=False, type=bool, help='Need to resize the input for fusion')
        self.parser.add_argument('--crop_before', default=[0, 0, 0, 0], type=Tuple[int, int, int, int], help='Need to crop the input image') #[35, 105, 25, 95]
        self.parser.add_argument('--crop_after', default=[0, 0, 0, 0], type=Tuple[int, int, int, int], help='Need to crop the aligned image') #[35, 105, 25, 95]
        self.parser.add_argument('--crop_common_roi', default=False, type=bool, help='Need to crop the aligned image automatically to common ROI') #[35, 105, 25, 95]
        self.parser.add_argument('--load_16bits', default=False, type=bool, help='Load 16 bits image if available')
        self.parser.add_argument('--loadSize', type=tuple, default=(480, 640), help='scale images to this size in dataloader')
        self.parser.add_argument('--inferSize', type=tuple, default=(480, 640), help='scale images to this size before fusion')
        self.parser.add_argument('--metrics', type=str or list, default=[], choices=list(METRICS_DICT.keys()), help='Compute metrics on the fused image')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.name is None:
            if self.opt.task == 'fusion':
                self.opt.name = list((FUSION_METHODS | WRAPPING_AND_FUSION_METHODS).keys())
            elif self.opt.task == 'wrapping':
                self.opt.name = WRAPPING_METHODS
            elif self.opt.task == 'wrapping_and_fusion':
                self.opt.name = WRAPPING_AND_FUSION_METHODS
            else:
                self.opt.name = All_METHODS
        else:
            self.opt.name = [self.opt.name] if isinstance(self.opt.name, str) or isinstance(self.opt.name, tuple) else self.opt.name
            if self.opt.task == 'fusion':
                for i, n in enumerate(self.opt.name):
                    if n.lower() == 'all':
                        self.opt.name.pop(i)
                        self.opt.name += list((FUSION_METHODS | WRAPPING_AND_FUSION_METHODS).keys())
                        continue
                    assert n.lower() in FUSION_METHODS | WRAPPING_AND_FUSION_METHODS, \
                        (f"Method {n} is not a fusion method. Please choose a method in "
                         f"{(FUSION_METHODS | WRAPPING_AND_FUSION_METHODS).keys()}")
                    self.opt.name[i] = n.lower()
            elif self.opt.task == 'wrapping':
                for i, n in enumerate(self.opt.name):
                    if n.lower() == 'all':
                        self.opt.name.pop(i)
                        self.opt.name += list((WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS).keys())
                        continue
                    assert n.lower() in WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS, \
                        (f"Method {n[0]} is not a wrapping method. Please choose a method in "
                         f"{(WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS).keys()}")
                    self.opt.name[i] = n.lower()
            elif self.opt.task == 'wrapping_and_fusion':
                for i, n in enumerate(self.opt.name):
                    if not isinstance(n, tuple):
                        if n.lower() == 'all':
                            self.opt.name.pop(i)
                            self.opt.name += WRAPPING_AND_FUSION_METHODS
                            continue
                        assert n.lower() in WRAPPING_AND_FUSION_METHODS, \
                            (f"Method {n} is not a wrapping_and_fusion method. Please choose a method in "
                             f"{WRAPPING_AND_FUSION_METHODS.keys()}")
                        self.opt.name[i] = n.lower()
                    else:
                        if n[0].lower() == 'all':
                            if n[1].lower() == 'all':
                                self.opt.name.pop(i)
                                self.opt.name += [(w, f) for w in WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS for
                                                  f in FUSION_METHODS + WRAPPING_AND_FUSION_METHODS]
                            else:
                                self.opt.name.pop(i)
                                self.opt.name += [(w, n[1].lower()) for w in WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS]
                        elif n[1].lower() == 'all':
                            self.opt.name.pop(i)
                            self.opt.name += [(n[0].lower(), f) for f in FUSION_METHODS | WRAPPING_AND_FUSION_METHODS]
                        else:
                            assert n[0].lower() in WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS, \
                                (f"Method {n[0]} is not a wrapping method. Please choose a method in "
                                 f"{(WRAPPING_METHODS | WRAPPING_AND_FUSION_METHODS).keys()}")
                            assert n[1].lower() in FUSION_METHODS | WRAPPING_AND_FUSION_METHODS, \
                                (f"Method {n[1]} is not a fusion method. Please choose a method in "
                                 f"{(FUSION_METHODS | WRAPPING_AND_FUSION_METHODS).keys()}")
                            self.opt.name[i] = (n[0].lower(), n[1].lower())
            else:
                for i, n in enumerate(self.opt.name):
                    if n.lower() == 'all':
                        self.opt.name.pop(i)
                        self.opt.name += list(All_METHODS.values())
                        continue
                    assert n.lower() in All_METHODS, \
                        (f"Method {n} is not a valid method. Please choose a method in "
                         f"{All_METHODS.keys()}")
                    self.opt.name[i] = n
                self.opt.metrics = []

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(os.getcwd(), 'fusion_framework/options')
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
