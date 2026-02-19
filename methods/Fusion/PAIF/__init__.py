import os
from collections import namedtuple
import torch
from ImagesCameras import ImageTensor

from .core.model_fusion_auto import Network_MM_Searched


def get_model(device, opt, **kwargs):
    Genotype = namedtuple('Genotype', 'normal_1 normal_1_concat normal_2 normal_2_concat normal_3 normal_3_concat')
    fusion_at = Genotype(normal_1=[('Denseblocks_3_1', 0), ('DilConv_3_2', 1)], normal_1_concat=[1, 2],
                         normal_2=[('Denseblocks_3_1', 0), ('Denseblocks_3_1', 1)], normal_2_concat=[1, 2],
                         normal_3=[('ECAattention_3', 0), ('Residualblocks_7_1', 1)], normal_3_concat=[1, 2])
    fusion_model_path = os.getcwd() + '/methods/Fusion/PAIF/checkpoints/model_meta30000_fusion_8.pth'
    model = Network_MM_Searched(32, fusion_at, None, None, 'mit_b3', num_classes=9)
    model.load_state_dict(torch.load(fusion_model_path, map_location='cpu'), strict=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model.to(device)

        @torch.no_grad()
        def forward(self, img_vis, img_ir):
            img_v_Y, img_v_Cb, img_v_Cr = img_vis.YCbCr().split(1, 1)
            fused_img, seg_map = self.model(img_ir, img_vis)
            fused_img = ImageTensor(torch.cat([fused_img.mean(dim=1, keepdim=True), img_v_Cb, img_v_Cr], dim=1), colorspace='YCbCr').RGB()
            return fused_img

    return Model()
