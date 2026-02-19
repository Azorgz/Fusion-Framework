import os

import cv2
import numpy as np
import torch
from ImagesCameras import ImageTensor
from torch import nn
from .model.SemLA import SemLA
from .model.utils import YCbCr2RGB


def get_model(device, opt, task=None, **kwargs):
    model = SemLA()
    reg_weight_path = "/methods/Wrapping_and_Fusion/SemLA/checkpoints/reg.ckpt"
    fusion_weight_path = "/methods/Wrapping_and_Fusion/SemLA/checkpoints/fusion75epoch.ckpt"
    # Loading the weights of the registration model
    model.load_state_dict(torch.load(os.getcwd() + reg_weight_path, map_location='cpu'), strict=False)

    # Loading the weights of the fusion model
    model.load_state_dict(torch.load(os.getcwd() + fusion_weight_path, map_location='cpu'), strict=False)

    model.eval()
    model = model.to(device)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.direction = opt.direction
            self.model = model
            self.task = task
            #  input size should be divisible by 8
            self.match_mode = 'scene'  # 'semantic' or 'scene'

        def forward(self, img_vis, img_ir):
            size = img_vis.shape[-2:]
            if size[0] % 8 != 0:
                h = (size[0] // 8) * 8
            else:
                h = size[0]
            if size[1] % 8 != 0:
                w = (size[1] // 8) * 8
            else:
                w = size[1]
            img_vis = img_vis.resize((h, w))
            img_ir = img_ir.resize((h, w)).GRAY()
            Y, Cb, Cr = img_vis.YCbCr().split(1, dim=1)
            vis_numpy = np.uint8(Y.squeeze().cpu().numpy() * 255)
            ir_numpy = img_ir.to_numpy().squeeze() * 255

            mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir = self.model(Y, img_ir.to_tensor(), matchmode=self.match_mode)

            if task != 'fusion':
                mkpts0 = mkpts0.cpu().numpy()
                mkpts1 = mkpts1.cpu().numpy()
                if self.direction == 'ir2vis':
                    _, prediction = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5)
                else:
                    _, prediction = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5)
                prediction = np.array(prediction, dtype=bool).reshape([-1])
                mkpts0_tps = mkpts0[prediction]
                mkpts1_tps = mkpts1[prediction]
                tps = cv2.createThinPlateSplineShapeTransformer()
                mkpts0_tps = mkpts0_tps.reshape(1, -1, 2)
                mkpts1_tps = mkpts1_tps.reshape(1, -1, 2)

                matches = []
                for j in range(1, mkpts0.shape[0] + 1):
                    matches.append(cv2.DMatch(j, j, 0))

                tps.estimateTransformation(mkpts0_tps, mkpts1_tps, matches)
                if self.direction == 'ir2vis':
                    ir_numpy = tps.warpImage(ir_numpy)
                    img_ir = (torch.from_numpy(ir_numpy / 255.)[None][None]).to(device)
                    sa_ir = tps.warpImage(sa_ir[0][0].detach().cpu().numpy())
                    sa_ir = torch.from_numpy(sa_ir)[None][None].to(device)
                else:
                    vis_numpy = tps.warpImage(vis_numpy)
                    Y = (torch.from_numpy(vis_numpy / 255.)[None][None]).to(device)
                    sa_ir = sa_ir[0][0].detach().cpu().numpy()
                    sa_ir = torch.from_numpy(sa_ir)[None][None].cuda()
            if self.task == 'wrapping':
                if self.direction == 'ir2vis':
                    return ImageTensor(ir_numpy)
                else:
                    return ImageTensor(vis_numpy)
            Y, Cb, Cr = img_vis.YCbCr().split(1, dim=1)
            fuse = self.model.fusion(torch.cat((Y, img_ir), dim=0), sa_ir, matchmode=self.match_mode).detach()
            fuse = YCbCr2RGB(fuse, Cb, Cr)
            return fuse

    model = Model()
    model.eval()
    return model
