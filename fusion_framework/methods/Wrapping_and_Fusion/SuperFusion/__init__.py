import os

from torch import nn

from .model import SuperFusion

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model(device, opt, task=None, **kwargs):
    model = SuperFusion(opt)
    model.resume(ROOT_DIR + '/checkpoints/RoadScene.pth', train=False)
    model = model.to(device)
    model.eval()

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model
            self.task = task
            self.direction = opt.direction

        def forward(self, img_vis, img_ir, img_ir_16bits=None):
            if self.task == 'wrapping':
                if img_ir_16bits is not None:
                    ret, ret_16bits = self.model.registration_forward(img_ir, img_vis, img_ir_16bits)
                    return (ret, img_ir, ret_16bits) if self.direction == 'vis2ir' else (img_vis, ret, ret_16bits)
                else:
                    ret = self.model.registration_forward(img_ir, img_vis)
                return (ret, img_ir) if self.direction == 'vis2ir' else (img_vis, ret)
            elif self.task == 'fusion':
                return self.model.fusion_forward(img_ir, img_vis), img_vis, img_ir
            else:
                ret = self.model.registration_forward(img_ir, img_vis)
                fus = self.model.fusion_forward(ret, img_vis) if self.direction == 'ir2vis' else self.model.fusion_forward(img_ir, ret)
                return fus, img_vis if self.direction == 'ir2vis' else ret, ret if self.direction == 'ir2vis' else img_ir

    return Model()
