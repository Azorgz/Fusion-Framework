import os

import torch
from torch import nn

from fusion_framework.methods.Wrapping.CrossRAFT.models.basic_blocks import back_warp
from fusion_framework.methods.Wrapping.CrossRAFT.models.cross_raft import CrossRAFT

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_model(device, opt, **kwargs):
    model = CrossRAFT(adapter=True)
    state_dict = torch.load(ROOT_DIR + '/checkpoints/checkpoint-10000.ckpt', weights_only=True)['state_dict']
    model.load_state_dict(state_dict)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.direction = opt.direction
            self.model = model.eval().to(device)
            self.ST = back_warp

        def forward(self, img_vis, img_ir):
            if img_vis.shape[-1] % 8 or img_vis.shape[-2] % 8:
                ori_shape = img_vis.shape[-2:]
                new_shape = (img_vis.shape[-2] // 8 * 8, img_vis.shape[-1] // 8 * 8)
                img_vis = img_vis.resize(new_shape)
                img_ir = img_ir.resize(new_shape)
            else:
                ori_shape = None
            if self.direction == 'ir2vis':
                img_tgt, img_src = img_vis, img_ir
            else:
                img_src, img_tgt = img_vis, img_ir
            flow = self.model(img_tgt, img_src)['flow']
            img_src = self.ST(img_src, flow)
            if ori_shape is not None:
                img_src = img_src.resize(ori_shape)
                img_tgt = img_tgt.resize(ori_shape)
            return (img_tgt, img_src) if self.direction == 'ir2vis' else (img_src, img_tgt)

    return Model()
