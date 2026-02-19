import os
from collections import namedtuple
from pathlib import Path

import torch
import yaml
from ImagesCameras import ImageTensor

from .config import from_dict
from .module.fuse.generator import Generator


def get_model(device, opt, **kwargs):
    config = from_dict(yaml.safe_load(Path('methods/Fusion/TarDAL/config/default.yaml').open('r')))
    f_dim, f_depth = config.fuse.dim, config.fuse.depth

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.generator = Generator(dim=f_dim, depth=f_depth).to(device)
            self.load_ckpt(torch.load(os.getcwd() + '/methods/Fusion/TarDAL/checkpoints/tardal-dt.pth', map_location='cpu'))

        def load_ckpt(self, ckpt: dict):
            f_ckpt = ckpt if 'fuse' not in ckpt else ckpt['fuse']
            if 'use_eval' in f_ckpt:
                f_ckpt.pop('use_eval')
            # load state dict
            self.generator.load_state_dict(f_ckpt)

        @torch.no_grad()
        def forward(self, img_vis, img_ir):
            vi_Y, vi_Cb, vi_Cr = img_vis.YCbCr().split(1, dim=1)
            fused_gray = self.generator(img_ir.GRAY(), vi_Y)
            fused_img = ImageTensor(torch.cat((fused_gray, vi_Cb, vi_Cr), dim=1), colorspace='YCbCr').RGB()
            return fused_img

    return Model()
