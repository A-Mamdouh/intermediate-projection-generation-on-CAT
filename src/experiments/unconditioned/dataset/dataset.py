import os
from typing import Tuple
from torch.utils.data import Dataset
import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from .config import DatasetConfig


class Dataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self._cfg = cfg
        self._object_dirs = os.listdir(cfg.data_dir)
        # Calculate some values that are handy for index and length calcuations
        self._num_objects = len(self._object_dirs)
        self._delta = cfg.delta
        self._sample_interval = cfg.sample_interval
        self._max_angle = cfg.max_angle - self._delta
        self._angles_per_object = int(self._max_angle / self._sample_interval)

    def __len__(self) -> int:
        """Returns the number of different inputs to the model"""
        return self._num_objects * self._angles_per_object

    def _get_img(self, object_index: int, angle: int) -> torch.Tensor:
        """Load image from memory and preprocess if needed"""
        img_path = os.path.join(
            self._cfg.data_dir, self._object_dirs[object_index], f"{angle}.png"
        )
        img = Image.open(img_path).resize(self._cfg.input_size)
        img_tensor = pil_to_tensor(img)
        if self._cfg.normalize:
            img_tensor = img_tensor / 255
        return img_tensor

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two images for input and a middle image for the target"""
        object_index = int(index / self._angles_per_object)
        angle_index = index % self._angles_per_object

        first_angle = angle_index * self._sample_interval
        second_angle = first_angle + self._delta
        target_angle = int((first_angle + second_angle) / 2)

        first_img = self._get_img(object_index, first_angle)
        second_img = self._get_img(object_index, second_angle)
        target_img = self._get_img(object_index, target_angle)

        x = torch.cat([first_img, second_img], 0)
        return x, target_img
