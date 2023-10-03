from dataclasses import dataclass
from src.utils.base_config import BaseConfig
from enum import Enum
from typing import Optional


class Losses(Enum):
    """Enum of supported losses"""

    MSE = "mse"
    L1 = "l1"
    CUSTOM_L1 = "custom_l1"
    CUSTOM_MSE = "custom_mse"


@dataclass
class TrainerConfig(BaseConfig):
    # Optimizer learning rate
    lr: float = 5e-4
    # Loss function for training
    loss: Losses = Losses.MSE
    epochs: int = 500
    # Batch size. If set to None, then a custom script runs to figure out the maximum possible batch size
    batch_size_train: Optional[int] = None
    batch_size_val: Optional[int] = None
    # early stopping to stop training when loss doesn't change / gets higher. Not used if set to None
    early_stopping_patience: Optional[int] = None
    use_validation: bool = True
    viz_sample_frequency: int = 4
    vis_sample_num: int = 2
    out_path: str = "./output"
    devices: int = 1
    progress_bar: bool = True
    resume: Optional[str] = None

    @staticmethod
    def from_dict(raw_dict) -> "TrainerConfig":
        parsed = {**raw_dict}
        if parsed.get("loss"):
            parsed["loss"] = Losses(raw_dict["loss"].strip().lower())
        return TrainerConfig(**parsed)