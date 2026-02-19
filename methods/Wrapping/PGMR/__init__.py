import torch
from ImagesCameras import ImageTensor
from torch import nn

from .code_pgmr.model import PGMR
from .enhance.swinir_arch import Resnet
from .options import MyOptions


def get_model(device, opt, **kwargs):
    parser = MyOptions()
    opts = parser.parse()
    model = PGMR(opts, device=device, direction=opt.direction)
    model = model.to(device)
    model.resume(opts.resume)
    model.eval()

    enhancer = Resnet()
    checkpoint = torch.load(opts.resume_enhancer)
    enhancer.load_state_dict(checkpoint["params"])
    enhancer = enhancer.to(device)
    enhancer.eval()

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.direction = opt.direction
            self.model = model
            self.enhancer = enhancer
            self.min_size = (128, 128)

        def forward(self, img_vis, img_ir):
            if img_vis.shape[-1] != img_vis.shape[-2]:
                old_shape = img_vis.shape[-2:]
                new_shape = tuple([max(img_vis.shape[-1], img_vis.shape[-2])])*2
                img_vis = img_vis.resize(new_shape).to_tensor()
                img_ir = img_ir.resize(new_shape).to_tensor()
                res = ImageTensor(self.enhancer(self.model.registration_forward(img_ir, img_vis)[0]).clip(0, 1)).resize(old_shape)
                if self.direction == 'ir2vis':
                    res = res.GRAY()
            else:
                res = ImageTensor(self.enhancer(self.model.registration_forward(img_ir, img_vis)[0]))
                if self.direction == 'ir2vis':
                    res = res.GRAY()
            return res

    model = Model()
    model.eval()
    return model
