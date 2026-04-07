import os
import torch
import sys

from ImagesCameras import ImageTensor

from fusion_framework.methods.Fusion.NightToDay.NightToday.NTIR2Day import NightToDay
from fusion_framework.methods.Fusion.NightToDay import NightToday
sys.modules["NightToday"] = NightToday


def get_model(device, opt, **kwargs):
    model = NightToDay('/home/godeta/PycharmProjects/MyTransform/checkpoints/download/weights/latest_net_NightToDay_UResNet', trainable=False)
    # model = NightToDay('/home/godeta/PycharmProjects/MyTransform/checkpoints/NightToday/100_net_ResNet', trainable=False)
    # model = NightToDay(os.getcwd() + '/fusion_framework/methods/Fusion/NightToDay/checkpoints/latest_net_NightToDay_IAware', trainable=False)
    # model = NightToDay(os.getcwd() + '/fusion_framework/methods/Fusion/NightToDay/checkpoints/latest_net_NightToDay_UResNet', trainable=False)

    model.opt.model.fusion_first = False

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model
            self.scale = 2**model.opt.model.gen.downscaling
            self.max_size = (1024, 1024)

        def forward(self, img_vis, img_ir):
            if img_vis.shape[-2] > self.max_size[0] or img_vis.shape[-1] > self.max_size[1]:
                ori_shape = img_vis.shape[-2:]
                scale_factor = min(self.max_size[0] / img_vis.shape[-2], self.max_size[1] / img_vis.shape[-1])
                new_size = (int(img_vis.shape[-2] * scale_factor), int(img_vis.shape[-1] * scale_factor))
                img_vis = img_vis.resize(new_size)
                img_ir = img_ir.resize(new_size)
            else:
                ori_shape = None
            if img_ir.shape[-1] % self.scale or img_ir.shape[-2] % self.scale:
                beforepad_shape = img_ir.shape[-2:]
                new_shape = ((img_ir.shape[-2] // self.scale + 1) * self.scale if img_ir.shape[-2] % self.scale else img_ir.shape[-2],
                             (img_ir.shape[-1] // self.scale + 1) * self.scale if img_ir.shape[-1] % self.scale else img_ir.shape[-1])
                pad = 0, new_shape[1] - img_ir.shape[-1], 0, new_shape[0] - img_ir.shape[-2]
                img_vis = img_vis.pad(pad)
                img_ir = img_ir.pad(pad)
            else:
                beforepad_shape = None
            with torch.no_grad():
                # fake_D = self.model(img_ir.to(self.device), align_first=False)
                img_vis[(img_vis.mean(1, keepdim=True) == 0).repeat(1, 3, 1, 1)] = 0.5
                # img_vis = img_vis*0.
                fake_D, fused_IR = self.model(img_ir.to(self.device), img_vis.to(self.device), return_fused_IR=True, align_first=False)
            if beforepad_shape is not None:
                fake_D = fake_D[..., :beforepad_shape[0], :beforepad_shape[1]]
                img_vis = img_vis.unpad()
                img_ir = img_ir.unpad()
            if ori_shape is not None:
                fake_D = ImageTensor(fake_D).resize(ori_shape)
                img_vis = img_vis.resize(ori_shape)
                img_ir = img_ir.resize(ori_shape)
            fake_D = (fake_D - fake_D.min()) / (fake_D.max() - fake_D.min() + 1e-8)
            return fake_D, img_vis, img_ir
    return Model()