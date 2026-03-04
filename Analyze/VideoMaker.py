import os
from os.path import isdir
from pathlib import Path
from typing import Tuple

import cv2
import imageio
import numpy as np
from ImagesCameras import ImageTensor
from imageio.core import Format
from tqdm import tqdm


def find_grid_shape(images):
    n = len(images)
    i, j = 0, 0
    while i * j < n:
        if i <= j:
            i += 1
        else:
            j += 1
    return i, j


class VideoMaker:
    def __init__(self, video_path=None, fps=None, colormap='gray'):
        assert isdir(video_path), f"Provided video path '{video_path}' is not a valid directory."
        self.video_path = video_path
        self.fps = fps
        self.colormap = colormap

    def __call__(self, images, name, fps=30, colormap=None, grid: Tuple | str = 'auto', start_frame=0, end_frame=-1):
        if colormap is not None:
            self.colormap = colormap
        assert start_frame >= 0, "Start frame must be non-negative."
        assert end_frame == -1 or all(len(os.listdir(path)) >= end_frame for path in images.values() if path is not None), "End frame exceeds the number of available images in one of the directories."
        if end_frame != -1:
            assert start_frame < end_frame, "Start frame must be less than end frame."
        imgs_list = {key: [path + p for p in sorted(os.listdir(path))[start_frame:end_frame]] for key, path in images.items() if path is not None}
        writer, length, size = self.create_video(imgs_list, name, fps=fps)
        assert all(len(imgs) == length for imgs in imgs_list.values())
        for _ in tqdm(range(length), desc=f'{name}.mp4'):
            imgs_input = {key: imgs.pop(0) for key, imgs in imgs_list.items()}
            composed_image = self._compose(size, grid=grid, **imgs_input)[..., [2, 1, 0]]
            writer.append_data(composed_image)
        writer.close()

    def create_video(self, imgs_list, name, fps=30):
        length = len(list(imgs_list.values())[0])
        video_name = self.video_path + f'{name}.mp4'
        sizes = [cv2.imread(imgs[0]).shape[:2] for imgs in imgs_list.values()]
        size = min(sizes, key=lambda x: x[0] * x[1])
        writer = imageio.get_writer(video_name, format="FFMPEG", mode="I", fps=fps)
        size = (int(size[0]), int(size[1]))
        return writer, length, size

    def _compose(self, size, grid: Tuple | str, **images):
        if len(images) == 0:
            raise ValueError("No images provided for composition.")
        elif len(images) == 1:
            return self._create_image(size, **images)
        elif len(images) > 1:
            return self._compose_grid(size, grid, **images)

    def _create_image(self, size, name, image):
        if image is None:
            im = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        else:
            im = ImageTensor(image).RGB(self.colormap).resize(size).to_opencv()
            # im = cv2.putText(im, name, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
            #                  (0, 0, 0), 1, cv2.LINE_AA)
        return im

    def _compose_grid(self, size, grid, **images):
        if grid == 'auto':
            grid = find_grid_shape(images)
        else:
            assert isinstance(grid, tuple) and len(grid) == 2, "Grid must be a tuple of (cols, rows)."
            assert grid[0] * grid[1] >= len(images), "Grid size is too small for the number of images."
        if grid[0] * grid[1] > len(images):
            print(f"Warning: Grid size {grid} is larger than the number of images {len(images)}. Some grid cells will be empty.")
            for i in range(grid[0] * grid[1] - len(images)):
                images[f'Empty_{i}'] = None
        i, j = 0, 0
        rows = []
        composed_images = []
        for name, image in images.items():
            if i >= grid[0]:
                i = 0
                j += 1
                composed_images.append(cv2.hconcat(rows))
                rows = []
            rows.append(self._create_image(size, name, image))
            i += 1
            if j >= grid[1]:
                raise ValueError("Too many images for the grid.")
        if rows:
            composed_images.append(cv2.hconcat(rows))
        composed_image = cv2.vconcat(composed_images)
        return composed_image


if __name__ == "__main__":
    video_maker = VideoMaker(video_path=os.getcwd() + '/results/videos/')
    # src = ['vis', 'ir', 'SeaFusion', 'SaliencyFus', 'FoalGAN', 'Ours']
    # pathes = [f'/home/godeta/Bureau/selection sequence/seq7/vis/',
    #           f'/home/godeta/Bureau/selection sequence/seq7/ir/',
    #           f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_SeAFusion/exp7/',
    #           f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_SaliencyMaskedFusion/exp7/',
    #           f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_NightToDay/exp7/',
    #           f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_NightToDay/exp7_mine/']
    src = ['0', '1', '2', '3']
    pathes = ["/home/godeta/Documents/IEEE - Transaction Aurelien Godet/Images/qualitative results/Source/",
              # "/home/godeta/Documents/IEEE - Transaction Aurelien Godet/Images/qualitative results/RIFT/",
              "/home/godeta/Documents/IEEE - Transaction Aurelien Godet/Images/qualitative results/PGMR/",
              "/home/godeta/Documents/IEEE - Transaction Aurelien Godet/Images/qualitative results/CrossRAFT/",
              "/home/godeta/Documents/IEEE - Transaction Aurelien Godet/Images/qualitative results/ours/",
              ]
    video_maker({k: v for k, v in zip(src, pathes)}, name='qualitative', fps=1, colormap='twilight', grid=(4, 1))
