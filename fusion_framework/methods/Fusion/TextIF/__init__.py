import os
import clip
import torch
from ImagesCameras import ImageTensor

from .model.Text_IF_model import Text_IF as create_model

from torch import Tensor

# Only work for 640x640 images, fuse gray layer from visible YCrCb transformed and the infrared image


def get_model(device, opt, **kwargs):
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)
    model.load_state_dict(torch.load(os.getcwd() + '/fusion_framework/methods/Fusion/TextIF/pretrained_weights/text_fusion.pth', map_location='cpu')['model'])
    model.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model
            # self.line = 'This is the infrared and visible light image fusion task'
            self.line =  "In the context of infrared-visible image fusion, visible images are susceptible to extremely low light degradation."
            # self.line = "In the context of infrared-visible image fusion, we aim to enhance the pedestrian visibility"
            # self.line = "In the context of infrared-visible image fusion, we aim to enhance a lot the pedestrian detection capability"
            # self.line = ""

        @torch.no_grad()
        def forward(self, img_vis, img_ir):
            height, width = img_vis.shape[-2:]
            new_width = (width // 16) * 16
            new_height = (height // 16) * 16
            img_ir_ = img_ir.resize((new_height, new_width))
            img_vis_ = img_vis.resize((new_height, new_width))
            text = clip.tokenize(self.line).to(device)
            fused_img = model(img_vis_, img_ir_, text)
            fused_img = ImageTensor(fused_img).resize((height, width)).normalize()
            return fused_img, img_vis, img_ir

    return Model()
