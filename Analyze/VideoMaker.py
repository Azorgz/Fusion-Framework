import os

import cv2
from ImagesCameras import ImageTensor
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
    def __init__(self, video_path=None, fps=None):
        self.video_path = video_path
        self.fps = fps

    def __call__(self, images, name, fps=30):
        imgs_list = {key: [path + p for p in sorted(os.listdir(path))] for key, path in images.items() if path is not None}
        video, length, size = self.create_video(imgs_list, name, fps=fps)
        assert all(len(imgs) == length for imgs in imgs_list.values())
        for _ in tqdm(range(length), desc=f'{name}.mp4'):
            imgs_input = {key: imgs.pop(0) for key, imgs in imgs_list.items()}
            composed_image = self._compose(size, **imgs_input)
            video.write(composed_image)
        cv2.destroyAllWindows()
        video.release()

    def create_video(self, imgs_list, name, fps=30):
        length = len(list(imgs_list.values())[0])
        video_name = self.video_path + f'{name}.mp4'
        sizes = [cv2.imread(imgs[0]).shape[:2] for imgs in imgs_list.values()]
        size = min(sizes, key=lambda x: x[0] * x[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_name, fourcc, fps, (size[1]*3, size[0]*2))
        size = (int(size[0]), int(size[1]))
        return video, length, size

    def _compose(self, size, **images):
        if len(images) == 0:
            raise ValueError("No images provided for composition.")
        elif len(images) == 1:
            return self._create_image(size, **images)
        elif len(images) > 1:
            return self._compose_grid(size, **images)

    def _create_image(self, size, name, image):
        im = ImageTensor(image).RGB().resize(size).to_opencv()
        im = cv2.putText(im, name, (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                         (255, 255, 255), 1, cv2.LINE_AA)
        return im

    def _compose_grid(self, size, **images):
        grid = find_grid_shape(images)
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
    video_maker = VideoMaker(video_path='/results/videos/')
    src = ['vis', 'ir', 'SeaFusion', 'SaliencyFus', 'FoalGAN', 'Ours']
    pathes = [f'/home/godeta/Bureau/selection sequence/seq7/vis/',
              f'/home/godeta/Bureau/selection sequence/seq7/ir/',
              f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_SeAFusion/exp7/',
              f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_SaliencyMaskedFusion/exp7/',
              f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_NightToDay/exp7/',
              f'/home/godeta/PycharmProjects/FusionMethods/results/XCalib_NightToDay/exp7_mine/']
    video_maker({k: v for k, v in zip(src, pathes)}, name='exp7', fps=3)
