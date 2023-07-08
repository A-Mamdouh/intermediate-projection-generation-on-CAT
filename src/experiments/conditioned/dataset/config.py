from src.utils.base_config import BaseConfig
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetConfig(BaseConfig):
    # Image size
    input_size: Tuple[int, int] = (256, 256)
    # Path to data directory
    data_dir: str = "./data/train"
    # Angle between first and second input image
    delta: int = 45
    # Sample frequency (take an image every n samples)
    sample_interval: int = 1
    # Maximum angle to use for training (in case of symmetric dataset)
    max_angle: int = 180
    # Normalize data to 0-1 float tensors
    normalize: bool = True
    encoder_depth: int = 4

    @staticmethod
    def from_dict(raw_dict) -> "DatasetConfig":
        parsed = {**raw_dict}
        if raw_dict.get("input_size") is not None:
            parsed["input_size"] = tuple(
                int(x.strip()) for x in raw_dict["input_size"].split(",")
            )
        return DatasetConfig(**parsed)
