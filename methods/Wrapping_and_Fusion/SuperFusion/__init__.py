import os

from torch import nn

from .model import SuperFusion


def get_model(device, opt, task=None, **kwargs):
    model = SuperFusion(opt)
    model.resume(os.getcwd() + '/methods/Wrapping_and_Fusion/SuperFusion/checkpoints/RoadScene.pth', train=False)
    model = model.to(device)
    model.eval()

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model
            self.task = task
            self.direction = opt.direction

        def forward(self, img_vis, img_ir):
            if self.task == 'wrapping':
                return self.model.registration_forward(img_ir, img_vis)
            elif self.task == 'fusion':
                return self.model.fusion_forward(img_ir, img_vis)
            else:
                return self.model(img_ir, img_vis)

    return Model()
