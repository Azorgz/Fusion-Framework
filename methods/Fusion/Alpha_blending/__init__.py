import torch
from ImagesCameras import ImageTensor

def get_model(device, opt, **kwargs):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Model(torch.nn.Module):
        def __init__(self, colormap='twilight', alpha=0.5):
            super(Model, self).__init__()
            self.device = device
            self.model = lambda img_vis, img_ir: alpha * img_vis + (1-alpha) * img_ir
            self.colormap = colormap

        def forward(self, img_vis, img_ir):
            img_ir = ImageTensor(img_ir).RGB(self.colormap).to_tensor()
            return self.model(img_vis, img_ir)

    return Model()