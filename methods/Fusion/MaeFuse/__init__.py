import os

import numpy as np
import torch
from ImagesCameras import ImageTensor
from torch import Tensor

from methods.Fusion.MaeFuse import fusion
from methods.Fusion.MaeFuse.connect_net import ConnectModel_MAE_Fuion as ConnectModel_MAE_Fusion
from methods.Fusion.MaeFuse.test_fusion import prepare_model

# Only work for 640x640 images, fuse gray layer from visible YCrCb transformed and the infrared image


def get_model(device, opt, **kwargs):
    model = prepare_model(arch="mae_decoder_4_640")
    fusion_layer = fusion.cross_fusion(embed_dim=1024)
    checkpoint = torch.load(os.getcwd() + '/methods/Fusion/MaeFuse/checkpoints/final_new_60.pth',
                            map_location='cpu', weights_only=False)
    connect = ConnectModel_MAE_Fusion(model, fusion_layer)
    connect.load_state_dict(checkpoint['model'], strict=True)
    model = connect.to(device)
    model.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model
            self.imagenet_mean = Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
            self.imagenet_std = Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
            self.fixed_size = (640, 640)

        def forward(self, img_vis, img_ir):
            vi_Y, vi_Cb, vi_Cr = img_vis.resize(self.fixed_size).YCbCr().split(1, dim=1)
            vi_Y = (ImageTensor(vi_Y).RGB() - self.imagenet_mean) / self.imagenet_std
            ir = (img_ir.resize(self.fixed_size).RGB() - self.imagenet_mean) / self.imagenet_std
            result = model(vi_Y, ir)
            pred = connect.model1.unpatchify(result)
            pred = (pred * self.imagenet_std + self.imagenet_mean).clip(0, 1).mean(dim=1, keepdim=True)
            fused_img = ImageTensor(torch.cat((pred, vi_Cb, vi_Cr), dim=1), colorspace='YCbCr').RGB()
            return fused_img

    return Model()
