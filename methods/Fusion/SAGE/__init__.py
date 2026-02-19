import os
from collections import namedtuple
import torch
from ImagesCameras import ImageTensor

from methods.Fusion.SAGE.model_sub.model import Network


def get_model(device, opt, **kwargs):
    model = Network()
    model.load_state_dict(torch.load(os.getcwd() + '/methods/Fusion/SAGE/checkpoints/checkpoints.pt', weights_only=True))
    model.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model.to(device)

        @torch.no_grad()
        def forward(self, img_vis, img_ir):
            img_v_Y, img_v_Cb, img_v_Cr = img_vis.YCbCr().split(1, 1)
            output, _ = model(img_v_Y, img_ir.GRAY())
            fused_img = ImageTensor(torch.cat([output.mean(dim=1, keepdim=True), img_v_Cb, img_v_Cr], dim=1), colorspace='YCbCr').RGB()
            return fused_img

    return Model()
