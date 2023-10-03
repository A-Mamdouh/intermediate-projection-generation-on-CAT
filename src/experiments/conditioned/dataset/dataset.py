import os
from typing import Tuple
from torch.utils.data import Dataset as BaseDataset
import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from .config import DatasetConfig


class Dataset(BaseDataset):
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
        height, width = self._cfg.input_size
        self._label_size = int(height / (2**self._cfg.encoder_depth)), int(width / (2**self._cfg.encoder_depth))

    def __len__(self) -> int:
        """Returns the number of different inputs to the model"""
        return self._num_objects * self._angles_per_object * self._delta
    
    @property
    def n_items(self) -> int:
        """Returns the number of objects"""
        return self._num_objects
    
    def get_for_prediction(self, object_id: int, angle: int):
        left_anchor = int(angle / self._delta) * self._delta
        right_anchor = (left_anchor + self._delta) % 360
        left_img = self._get_img(object_id, left_anchor)
        right_img = self._get_img(object_id, right_anchor)
        target_img = self._get_img(object_id, angle)
        z = (angle % self._delta) / self._delta
        z = torch.ones((1, *self._label_size)) * z
        x = torch.cat([left_img, right_img], 0)
        return (x, z), target_img


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

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return two images for input and a middle image for the target"""
        object_index = int(index / (self._angles_per_object * self._delta))
        angle_index = int(index / self._delta) % self._angles_per_object

        first_angle = angle_index * self._sample_interval
        second_angle = first_angle + self._delta
        target_angle = first_angle + (index % self._delta)
        label = (index % self._delta) / self._delta
        z = torch.ones((1, *self._label_size)) * label
        second_img = self._get_img(object_index, second_angle)
        first_img = self._get_img(object_index, first_angle)
        target_img = self._get_img(object_index, target_angle)

        x = torch.cat([first_img, second_img], 0)
        return (x, z), target_img
