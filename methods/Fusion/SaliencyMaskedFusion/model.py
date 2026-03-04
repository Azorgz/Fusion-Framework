import numpy as np
import torch
from cv2 import CV_64F, getGaborKernel
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.filters import bilateral_blur
from kornia.geometry import ScalePyramid
from torch import nn
from torch.nn.functional import interpolate, conv2d
from torchvision.transforms.v2.functional import gaussian_blur

from methods.Fusion.SaliencyMaskedFusion.utils import get_GaborFilters


class SaliencyFuse(nn.Module):
    def __init__(self, device, inferSize, **kwargs):
        super(SaliencyFuse, self).__init__()
        self.device = device
        self.inference_size = inferSize
        self.c = [1, 2, 3]  # Central scales
        self.delta_intensity = [3, 4]  # Surrounding scales
        self.delta_orientation = [2, 3]  # Surrounding scales
        self.sigmas = np.arange(1, 5)  # Sigma values for Gabor filters
        self.gabor_filters = [get_GaborFilters(ksize=9, sigma=sigma,
                                               gamma=self.inference_size[0]/self.inference_size[1],
                                               device=device, dtype=torch.float32) for sigma in self.sigmas]

    def forward(self, img_vis, img_ir):
        img_vis, img_ir = img_vis.to(self.device), img_ir.to(self.device)
        # Resize visible image to infrared image size
        img_vis = interpolate(img_vis, size=img_ir.shape[-2:], mode='bilinear', align_corners=False)

        p_vis, *_ = self._get_gaussian_pyr(img_vis)
        p_ir, shape, B = self._get_gaussian_pyr(img_ir)

        # Intensity Saliency maps
        intensity_saliency_vis = self._compute_intensity_saliency(p_vis, shape)
        intensity_saliency_ir = self._compute_intensity_saliency(p_ir, shape)

        # Gabor Orientation Saliency maps
        orientation_saliency_vis = self._compute_orientation_saliency(p_vis, shape, B)
        orientation_saliency_ir = self._compute_orientation_saliency(p_ir, shape, B)

        initial_fusion = self._fusion_maps(intensity_saliency_vis, intensity_saliency_ir, orientation_saliency_vis, orientation_saliency_ir, img_vis, img_ir)
        details_enhanced_fusion = self._enhance_details(initial_fusion)

        ab = rgb_to_lab(img_vis)[:, 1:3, :, :]
        details_enhanced_fusion = torch.cat([details_enhanced_fusion*100, ab], dim=1)
        finally_fused_rgb = lab_to_rgb(details_enhanced_fusion)
        return finally_fused_rgb

    def _fusion_maps(self, intensity_saliency_vis, intensity_saliency_ir, orientation_saliency_vis, orientation_saliency_ir, img_vis, img_ir):
        """
        :param intensity_saliency_vis: (B, 1, H, W)
        :param intensity_saliency_ir: (B, 1, H, W)
        :param orientation_saliency_vis: (B, 1, H, W)
        :param orientation_saliency_ir: (B, 1, H, W)
        :return: feature_maps: (B, 2, H, W)
        """
        if img_vis.shape[1] == 3:
            img_vis = rgb_to_lab(img_vis)[:, 0:1]/100  # Use only L channel
        if img_ir.shape[1] == 3:
            img_ir = img_ir.mean(1, keepdim=True)
        # compute euclidian distance between features and real Images
        d_intensity_vis = torch.sqrt(((intensity_saliency_vis - img_vis)**2).sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
        d_intensity_ir = torch.sqrt(((intensity_saliency_ir - img_ir)**2).sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
        d_orientation_vis = torch.sqrt(((orientation_saliency_vis - img_vis)**2).sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
        d_orientation_ir = torch.sqrt(((orientation_saliency_ir - img_ir)**2).sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
        # compute Saliency maps
        SM_vis = (d_intensity_vis / (d_intensity_vis + d_orientation_vis) * orientation_saliency_vis +
                  d_orientation_vis / (d_intensity_vis + d_orientation_vis) * intensity_saliency_vis)
        SM_ir = (d_intensity_ir / (d_intensity_ir + d_orientation_ir) * orientation_saliency_ir +
                 d_orientation_ir / (d_intensity_ir + d_orientation_ir) * intensity_saliency_ir)
        # Initial Fusion maps
        initial_fusion_map_vis = SM_vis / (SM_vis + SM_ir + 1e-6) * img_vis
        initial_fusion_map_ir = SM_ir / (SM_vis + SM_ir + 1e-6) * img_ir
        initial_fusion_maps = (initial_fusion_map_vis - initial_fusion_map_vis.min()) / (initial_fusion_map_vis.max() - initial_fusion_map_vis.min() + 1e-6) + \
                              (initial_fusion_map_ir - initial_fusion_map_ir.min()) / (initial_fusion_map_ir.max() - initial_fusion_map_ir.min() + 1e-6)
        return initial_fusion_maps

    def _enhance_details(self, fusion_map):
        """
        :param fusion_map: (B, 1, H, W)
        :return: details_enhanced_fusion: (B, 1, H, W)
        """
        details = fusion_map - bilateral_blur(fusion_map, kernel_size=(5, 5), sigma_space=(1.6, 1.6), sigma_color=.1)
        details_enhanced_fusion = fusion_map + torch.log(details + 1) / 100
        return (details_enhanced_fusion - details_enhanced_fusion.min()) / (details_enhanced_fusion.max() - details_enhanced_fusion.min() + 1e-6)


    def _get_gaussian_pyr(self, img):
        """
        Build Gaussian Pyramid
        :param img: (B, C, H, W)
        :return: gaussian_pyr: list of (B, H_l, W_l, C)
        """
        B = img.shape[0]
        if img.shape[1] == 3:
            img = rgb_to_lab(img)[:, 0:1]/100  # Use only L channel
        shape = img.shape[-2:]
        min_size = max(shape[-2] // (2 ** 8), 1)
        sp = ScalePyramid(n_levels=1, init_sigma=1, min_size=min_size, double_image=False)
        gaussian_pyr, *_ = sp(img)
        return gaussian_pyr, shape, B

    def _compute_intensity_saliency(self, gaussian_pyr, shape):
        """
        Compute color saliency maps based on the method from
        "Color Saliency Detection Based on Human Color Perception" by Guo et al.
        """
        res = []
        for c in self.c:
            for delta in self.delta_intensity:
                # Reference image (B, H, W, 3)
                high = gaussian_pyr[c][:, :, 0]
                low = gaussian_pyr[c + delta][:, :, 0]
                blur = interpolate(low, size=high.shape[-2:], mode='bilinear', align_corners=False)
                diff = torch.abs(high - blur)
                res.append((diff - diff.min()) / (diff.max() - diff.min() + 1e-6))
        res = torch.stack([interpolate(r, shape) for r in res], dim=1).sum(dim=1)
        return (res - res.min()) / (res.max() - res.min() + 1e-6)

    def _compute_orientation_saliency(self, gaussian_pyr, shape, B):
        """
        :param img:
        :return:
        """
        gamma = shape[0] / shape[1]
        res = []

        for c in self.c:
            for delta in self.delta_orientation:
                # Reference image (B, H, W, 3)
                high = gaussian_pyr[c][:, :, 0]
                low = gaussian_pyr[c + delta][:, :, 0]
                gabor_kernel_high = self.gabor_filters[0]
                gabor_kernel_low = self.gabor_filters[self.delta_orientation.index(delta)]
                O_low = torch.cat([conv2d(low, gk.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1), padding=4) for gk in gabor_kernel_low], dim=1)
                O_high = torch.cat([conv2d(high, gk.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1), padding=4) for gk in gabor_kernel_high], dim=1)
                O_blur = interpolate(O_low, size=high.shape[-2:], mode='bilinear', align_corners=False)
                diff = torch.abs(O_high - O_blur)  # B, orientations, H, W
                diff_min = diff.min(dim=-1)[0].min(dim=-1)[0][..., None, None]
                diff_max = diff.max(dim=-1)[0].max(dim=-1)[0][..., None, None]
                diff_norm = (diff - diff_min)/(diff_max - diff_min + 1e-6)
                res.append(diff_norm)
        # sum for all deltas and centers
        res = torch.stack([interpolate(r, shape) for r in res], dim=1).sum(dim=1)
        res_min = res.min(dim=-1)[0].min(dim=-1)[0][..., None, None]
        res_max = res.max(dim=-1)[0].max(dim=-1)[0][..., None, None]
        res_norm = (res - res_min)/(res_max - res_min + 1e-6)
        # sum for all orientations
        res_final = res_norm.sum(dim=1, keepdim=True)
        return (res_final - res_final.min()) / (res_final.max() - res_final.min() + 1e-6)




        #         Rr = high[:, 0:1]
        #         Gr = high[:, 1:2]
        #         Br = high[:, 2:3]
        #
        #         Rc = Rr - (Gr + Br) / 2
        #         Gc = Gr - (Rr + Br) / 2
        #         Bc = Br - (Rr + Gr) / 2
        #         Yc = (Rr + Gr) / 2 - torch.abs(Rr - Gr) / 2 - Br
        #
        #         Rb = blur[:, 0]
        #         Gb = blur[:, 1]
        #         Bb = blur[:, 2]
        #
        #         Rs = Rb - (Gb + Bb) / 2
        #
        #         # Exact same equations
        #         RG = torch.abs((Rc - Gc) - (Gc - Rs))
        #         BY = torch.abs((Bc - Yc) - (Bc - Rs))
        #
        #         sal = 0.5 * RG + 0.5 * BY
        #
        #         # Gaussian filtering
        #         sal = gaussian_blur(sal, kernel_size=[3, 3])
        #
        #         RGBY[c][d] = sal
        #
        # return RGBY
