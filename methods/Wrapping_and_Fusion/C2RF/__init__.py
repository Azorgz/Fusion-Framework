import torch
import os

from torch import nn

from .model import C2RF
from .options import TrainOptions


def get_model(device, opt, task=None, **kwargs):
    parser = TrainOptions()
    opts = parser.parse()
    model_path = os.getcwd() + f'/methods/Wrapping_and_Fusion/C2RF/checkpoints/{opts.dataset}/'
    model = C2RF(opt)
    model.AE_encoder.load_state_dict(torch.load(os.path.join(model_path, 'Encoder.pth'), map_location='cpu'))
    model.AE_decoder.load_state_dict(torch.load(os.path.join(model_path, 'Decoder.pth'), map_location='cpu'))
    model.fusion_layer.load_state_dict(torch.load(os.path.join(model_path, 'Fusion_layer.pth'), map_location='cpu'))
    model.reg_net.load_state_dict(torch.load(os.path.join(model_path, 'RegNet.pth'), map_location='cpu'))
    model = model.to(device)
    model.eval()

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model
            self.task = task

        def forward(self, img_vis, img_ir):
            if self.task == 'wrapping':
                return self.model(img_vis, img_ir)[0]
            elif self.task == 'fusion':
                return self.model(img_vis, img_ir)[1]
            else:
                return self.model(img_vis, img_ir)[1]

    return Model()
