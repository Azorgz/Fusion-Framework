import torch

from methods.Fusion.SaliencyMaskedFusion.model import SaliencyFuse


def get_model(device, opt, **kwargs):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SaliencyFuse(device=device)
    model.eval()

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.device = device
            self.model = model

        def forward(self, img_vis, img_ir):
            return self.model(img_vis, img_ir)

    return Model()
