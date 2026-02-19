import os
import torch
import sys
sys.path.insert(0, "/home/godeta/PycharmProjects/FusionMethods/methods/Fusion/NightToDay")

from methods.Fusion.NightToDay.NightToday.NTIR2Day import Image2ImageGAT_Dual


def get_model(device, opt, **kwargs):
    # model = Image2ImageGAT_Dual('/home/godeta/PycharmProjects/FusionMethods/methods/Fusion/NightToDay/checkpoints/latest_net_ResNet_red_ped', trainable=False)
    model = Image2ImageGAT_Dual('/home/godeta/PycharmProjects/MyTransform/checkpoints/download/latest_net_ResNet', trainable=False)
    # model = Image2ImageGAT_Dual('/home/godeta/PycharmProjects/MyTransform/checkpoints/download/latest_net_ResNet', trainable=False)
    # model = Image2ImageGAT_Dual('/home/godeta/PycharmProjects/MyTransform/checkpoints/NightToday/100_net_ResNet', trainable=False)
    # model = Image2ImageGAT_Dual(os.getcwd() + '/methods/Fusion/NightToDay/checkpoints/latest_net_ResNet', trainable=False)
    model.opt.model.fusion_first = False

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model

        def forward(self, img_vis, img_ir):
            with torch.no_grad():
                # fake_D = self.model(img_ir.to(self.device), align_first=False)
                img_vis[(img_vis.mean(1, keepdim=True) == 0).repeat(1, 3, 1, 1)] = 0.5
                # img_vis = 0.5 + img_vis*0.
                fake_D, fused_IR = self.model(img_ir.to(self.device), img_vis.to(self.device), return_fused_IR=True, align_first=False)
            return fake_D

    return Model()
