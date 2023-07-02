from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class _ConfigBase(ABC):
    def to_dict(self) -> Dict[str, Any]:
        # Shamelessly copied from: https://stackoverflow.com/a/64693838
        def custom_asdict_factory(data):
            def convert_value(obj):
                if isinstance(obj, Enum):
                    return obj.value
                if isinstance(obj, Tuple):
                    return ", ".join(str(x) for x in obj)
                return obj

            return {key: convert_value(value) for key, value in data}

        return asdict(self, dict_factory=custom_asdict_factory)

    @abstractmethod
    def from_dict(raw_dict: Dict[str, Any]) -> "_ConfigBase":
        pass


class Activation(Enum):
    """Enum of supported activation functions"""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SILU = "silu"


class Losses(Enum):
    """Enum of supported losses"""

    MSE = "mse"
    L1 = "l1"
    CUSTOM_L1 = "custom_l1"
    CUSTOM_MSE = "custom_mse"


@dataclass
class ModelConfig:
    # Activation function used for hidden layers
    activation: "Activation" = Activation.LEAKY_RELU
    # Depth of the model
    depth: int = 4
    # Starting channels of the first encoder layer
    start_channels: int = 128
    # Factor of channel increase between layers
    dilation: float = 2.0
    # Use upsample layer for upsampling. Uses conv_transpose if False
    up_sample: bool = False
    # Use maxpooling layer for downsampling. Uses strided convolution if False
    maxpool: bool = False

    @staticmethod
    def from_dict(raw_dict) -> "ModelConfig":
        raw_dict["activation"] = Activation(raw_dict["activation"].strip().lower())
        return ModelConfig(**raw_dict)


@dataclass
class DatasetConfig(_ConfigBase):
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

    @staticmethod
    def from_dict(raw_dict) -> "DatasetConfig":
        raw_dict["input_size"] = tuple(
            int(x.strip()) for x in raw_dict["input_size"].split(",")
        )
        return DatasetConfig(**raw_dict)


@dataclass
class TrainerConfig(_ConfigBase):
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

    @staticmethod
    def from_dict(raw_dict) -> "TrainerConfig":
        raw_dict["loss"] = Losses(raw_dict["loss"].strip().lower())
        return TrainerConfig(**raw_dict)


@dataclass
class ExperimentConfig(_ConfigBase):
    """This class holds configuration details of the autoencoder experiments"""

    model: "ModelConfig" = field(default_factory=ModelConfig)
    dataset_train: "DatasetConfig" = field(default_factory=DatasetConfig)
    dataset_val: "DatasetConfig" = field(
        default_factory=lambda: DatasetConfig(data_dir="./data/val")
    )
    trainer: "TrainerConfig" = field(default_factory=TrainerConfig)
    # If true, copy missing validation dataset keys from train dataset
    val_copy_train: bool = True

    @staticmethod
    def from_dict(raw_dict) -> "ExperimentConfig":
        model_config = ModelConfig.from_dict(raw_dict["model"])
        dataset_config_train = DatasetConfig.from_dict(raw_dict["dataset_train"])
        trainer_config = TrainerConfig.from_dict(raw_dict["trainer"])
        if raw_dict["val_copy_train"]:
            dataset_config_val = DatasetConfig.from_dict(
                {**dataset_config_train.to_dict(), **raw_dict["dataset_val"]}
            )
        else:
            dataset_config_val = DatasetConfig.from_dict(raw_dict["dataset_val"])
        return ExperimentConfig(
            model_config, dataset_config_train, dataset_config_val, trainer_config
        )
