import os

import torch
from .scripts.calculate import load_lookup_table, Generator_for_info, apply_fusion_4d_with_interpolation


def get_model(device, opt, **kwargs):
    device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lut = load_lookup_table(os.getcwd() + '/methods/Fusion/LUT_Fuse/ckpts/fine_tuned_lut_original.npy').to(device)
    if lut is None:
        return

    get_context = Generator_for_info()
    get_context.load_state_dict(torch.load(os.getcwd() + '/methods/Fusion/LUT_Fuse/ckpts/generator_context_original.pth', map_location='cpu'))
    get_context = get_context.to(device)
    get_context.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.lut = lut
            self.context = get_context

        def forward(self, img_vis, img_ir):
            fused_result = apply_fusion_4d_with_interpolation(img_vis*255, img_ir.mean(dim=1, keepdim=True)*255, self.lut, self.context)
            return fused_result.clamp(0, 1)

    return Model()
