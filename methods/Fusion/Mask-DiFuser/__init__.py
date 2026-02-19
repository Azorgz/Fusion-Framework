import argparse

import numpy as np
import torch
from .Diffusion.diffusion import GaussianDiffusionSampler
from .Diffusion.model import UNet


def get_model(device, opt, **kwargs):
    parser = argparse.ArgumentParser()
    modelArgs = {
        "epoch": 3500,
        "batch_size": 2,
        "val_batch": 1,
        "num_workers": 8,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.0,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1.,
        "fluency": 500,
        "change_epoch": 100,
        "ddim": True,
        "ddim_step": 5,
    }

    parser.add_argument('--pretrained_path', type=str, default='methods/Fusion/Mask-DiFuser/checkpoints/model.pt', help='Path to the pretrained model')
    parser.add_argument('--task_type', type=str, default='VIF', help='options: VIF, MEF, MFF, Med, Pol, Nir')
    parser.add_argument("--seed", default=3407, type=int)
    args = parser.parse_args()

    for key, value in modelArgs.items():
        setattr(args, key, value)
    model = UNet(T=args.T, ch=args.channel, ch_mult=args.channel_mult, attn=args.attn,
                 num_res_blocks=args.num_res_blocks, dropout=0.)
    ckpt_path = args.pretrained_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    model.eval()
    sampler = GaussianDiffusionSampler(model, args.beta_1, args.beta_T, args.T)
    multiple = 2 ** len(args.channel_mult)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = torch.device(device)
            self.model = model.to(device)
            self.sampler = sampler.to(device)
            self.multiple = multiple

        def forward(self, img_vis, img_ir):
            _, _, h, w = img_ir.shape
            crop_height = int(self.multiple * np.ceil(h / self.multiple))
            crop_width = int(self.multiple * np.ceil(w / self.multiple))

            img_ir = img_ir.pad((0, crop_width - w, 0, crop_height - h), mode='reflect')
            img_vis = img_vis.pad((0, crop_width - w, 0, crop_height - h), mode='reflect')
            data_concate = torch.cat([img_ir, img_vis], dim=1)
            sampledImgs, _ = self.sampler(data_concate, ddim=True, ddim_step=args.ddim_step, ddim_eta=1.0,
                                             seed=args.seed, type=args.task_type)
            return self.normalize(sampledImgs[:, :, :h, :w])

        def normalize(self, x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

    return Model()


