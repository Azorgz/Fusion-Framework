import os
import torch
from methods.Fusion.SeAFusion.FusionNet import FusionNet
from methods.Fusion.SeAFusion.utils import YCbCr2RGB, RGB2YCrCb


def get_model(device, opt, **kwargs):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(output=1)
    model.load_state_dict(torch.load(os.getcwd() + '/methods/Fusion/SeAFusion/model/Fusion/fusionmodel_final.pth'))
    model = model.to(device)
    model.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model

        def forward(self, img_vis, img_ir):
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            fused_img = model(vi_Y, img_ir.GRAY())
            return YCbCr2RGB(fused_img, vi_Cb, vi_Cr)

    return Model()
