import os

import numpy as np
import torch
from ImagesCameras import ImageTensor
from kornia.filters import median_blur

from fusion_framework.datasets.DatasetBase import DatasetBase


class TestLightness(DatasetBase):
    """
    BaseDataset class for the Lightness Experiment.
    """
    root_dir = "/home/godeta/Bureau/selection sequence/test_lightness/"

    night_levels: int = 11
    temperature: int = 28  # in degrees Celsius, for the dark current noise generation
    exposure_time: float = 0.025  # in seconds, for the dark current noise generation (per default 1/fps)
    black_level_offset: float = 5.0  # in [0, 100], to simulate the black level offset of the sensor in % of the maximum pixel value
    full_well_capacity: float = 20000  # in electrons, for the dark current noise generation
    leaky_pixel_percentage: float = 0.000  # percentage of pixels that are 'leaky' (hot pixels) (%)
    median_filter_size: int = 3  # size of the median filter applied to the hot pixel map (must be odd)

    def __init__(self, opt):
        self.path_vis = self.root_dir + "vis/"
        self.path_ir = self.root_dir + "ir/"
        self.path_degraded = self.root_dir + "noisy/"
        self.noise_sigma_per_channel: tuple[float, float, float] = (0.0380987636744976, 0.0388190858066082, 0.0499677807092667)
        self.noise_mean_per_channel: tuple[float, float, float] = (0, 0, 0)
        self.night_scale = torch.arange(0, self.night_levels) / (self.night_levels - 1)  # from 0 to 1
        self.hot_pixel_map = None  # Initialize hot pixel map as None
        super().__init__(opt)
        self.list_img_degraded = [self.path_degraded + f for f in sorted(os.listdir(self.path_degraded))]

    def __len__(self):
        return len(self.image_vis) * self.night_levels

    def __getitem__(self, index):
        idx = index // self.night_levels
        night_level = self.night_scale[index % self.night_levels]
        img_vis, img_ir, shape = super().__getitem__(idx)
        if self.path_degraded + img_vis.name + f"_{int(night_level*100):03d}%_luminance.png" in self.list_img_degraded:
            img_vis_noised = ImageTensor(self.path_degraded + img_vis.name + f"_{int(night_level*100):03d}%_luminance.png")
        else:
            img_vis_noised = self._process_day(img_vis, night_level)
            img_vis_noised.name = img_vis.name + f"_{int(night_level*100):03d}%_luminance"
            img_vis_noised.save(self.path_degraded, depth=8)
            img_vis_noised = ImageTensor(self.path_degraded + img_vis_noised.name + ".png")
        if self.direction != 'ir2vis':
            img_vis_noised = img_vis_noised.match_shape(img_ir)
        else:
            img_ir = img_ir.match_shape(img_vis_noised)
        return img_vis_noised, img_ir, shape

    def _process_day(self, img_vis, night_level):
        # decrease luminance of the visible image according to the night level
        img_vis_night = img_vis * night_level
        shape = img_vis_night.shape[-2:]
        gaussian_noise = self._generate_gaussian_noise(shape)
        dark_noise = self._generate_dark_current_noise(shape)
        offset = self.black_level_offset / 100.0
        noise_image = dark_noise + gaussian_noise + offset
        img_vis_night_noisy = (img_vis_night * (1 - offset) + noise_image).clamp(0, 1)
        return ImageTensor(img_vis_night_noisy)

    def _generate_gaussian_noise(self, shape):
        noise_r = torch.randn(shape) * self.noise_sigma_per_channel[0] + self.noise_mean_per_channel[0]
        noise_g = torch.randn(shape) * self.noise_sigma_per_channel[1] + self.noise_mean_per_channel[1]
        noise_b = torch.randn(shape) * self.noise_sigma_per_channel[2] + self.noise_mean_per_channel[2]
        gaussian_noise = torch.stack((noise_r, noise_g, noise_b), dim=0)
        return gaussian_noise

    def _generate_dark_current_noise(self, shape):
        # 1. Base dark current (Poisson)
        mean_electrons = self._estimate_dark_rate() * self.exposure_time
        dark_shot_noise = torch.poisson(torch.full((1, *shape), mean_electrons))
        # 2. Add Hot Pixels (DSNU)
        # We simulate a few pixels that are 100x leakier
        if self.hot_pixel_map is None and self.leaky_pixel_percentage > 0:
            self.hot_pixel_map = self._generate_hot_pixels(shape)
        else:
            self.hot_pixel_map = torch.zeros(shape)
        hot_pixel_noise = self.hot_pixel_map * (mean_electrons * 50)
        total_thermal = (dark_shot_noise + hot_pixel_noise).repeat(3, 1, 1)
        return total_thermal / self.full_well_capacity

    def _generate_hot_pixels(self, shape):
        # Create a static mask of 'leaky' pixels (% of pixels)
        indices = torch.rand(shape) > 1 - self.leaky_pixel_percentage/100
        hot_pixel_map = torch.zeros(shape)
        # Hot pixels leak significantly more electrons
        hot_pixel_map[indices] = torch.rand(indices.sum()) * 0.5
        return hot_pixel_map

    def _estimate_dark_rate(self):
        return 2.0 * 2**((self.temperature - 20) / 8)

