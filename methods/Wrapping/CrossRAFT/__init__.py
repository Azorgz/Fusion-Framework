import os

import torch
from torch import nn

from methods.Wrapping.CrossRAFT.models.basic_blocks import back_warp
from methods.Wrapping.CrossRAFT.models.cross_raft import CrossRAFT


def get_model(device, opt, **kwargs):
    model = CrossRAFT(adapter=True)
    state_dict = torch.load(os.getcwd() + '/methods/Wrapping/CrossRAFT/checkpoints/checkpoint-10000.ckpt',
                            weights_only=True)['state_dict']
    model.load_state_dict(state_dict)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.direction = opt.direction
            self.model = model.eval().to(device)
            self.ST = back_warp

        def forward(self, img_vis, img_ir):
            if self.direction == 'ir2vis':
                img_tgt, img_src = img_vis, img_ir
            else:
                img_tgt, img_src = img_ir, img_vis
            flow = self.model(img_tgt, img_src)['flow']
            return self.ST(img_src, flow)

    return Model()
