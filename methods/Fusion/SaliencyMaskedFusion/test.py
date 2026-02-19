from ImagesCameras import ImageTensor

from methods.Fusion.SaliencyMaskedFusion.model import SaliencyFuse

vis = ImageTensor('/home/godeta/PycharmProjects/TIR2VIS/datasets/LYNRED/LYNRED_datasets/trainC/image_02625.png').to('cuda')
ir = ImageTensor('/home/godeta/PycharmProjects/TIR2VIS/datasets/LYNRED/LYNRED_datasets/trainB/image_02625.png').to('cuda')
model = SaliencyFuse()

model(vis, ir)
