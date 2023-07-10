from dataclasses import dataclass
from enum import Enum
from ..unconditioned.autoencoder.config import ModelConfig as AutoencoderConfig
from ..unconditioned.unet.config import ModelConfig as UnetConfig
from ..unconditioned.dataset.config import DatasetConfig as UnconditionedDatsetConfig
from ..unconditioned.trainer.config import TrainerConfig as UnconditionedTrainerConfig
from ..conditioned.dataset.config import DatasetConfig as ConditionedDatasetConfig
from ..conditioned.trainer.config import TrainerConfig as ConditionedTrainerConfig
from ..conditioned.cunet.config import ModelConfig as CUnetConfig
from src.utils.base_config import BaseConfig
from typing import Dict, Tuple, Optional


class ExpType(Enum):
    AUTOENCODER = "autoencoder"
    UNET = "unet"
    CUNET = "cunet"


_exp_type_to_configs: Dict[ExpType, Tuple[BaseConfig, BaseConfig, BaseConfig]] = {
    ExpType.AUTOENCODER: (AutoencoderConfig, UnconditionedDatsetConfig, UnconditionedTrainerConfig),
    ExpType.UNET: (UnetConfig, UnconditionedDatsetConfig, UnconditionedTrainerConfig),
    ExpType.CUNET: (CUnetConfig, ConditionedDatasetConfig, ConditionedTrainerConfig)
}


@dataclass
class ExperimentConfig(BaseConfig):
    """This class holds configuration details of the experiment"""
    exp_type: ExpType
    model: BaseConfig
    dataset_train: BaseConfig
    dataset_val: BaseConfig
    trainer: BaseConfig
    accelerator: Optional[str] = "cpu"
    # If true, copy missing validation dataset keys from train dataset
    val_copy_train: bool = True

    @staticmethod
    def from_dict(raw_dict) -> "ExperimentConfig":
        config = {}
        config["exp_type"] = ExpType(raw_dict["exp_type"])
        ModelConfig, DatasetConfig, TrainerConfig = _exp_type_to_configs[config["exp_type"]]
        config["model"] = ModelConfig.from_dict(raw_dict["model"])
        config["dataset_train"] = DatasetConfig.from_dict(raw_dict["dataset_train"])
        config["val_copy_train"] = raw_dict["val_copy_train"]
        if config["val_copy_train"]:
            config["dataset_val"] = DatasetConfig.from_dict({**raw_dict["dataset_train"], **raw_dict["dataset_val"]})
        else:
            config["dataset_val"] = DatasetConfig.from_dict(raw_dict["dataset_val"])
        config["trainer"] = TrainerConfig.from_dict(raw_dict["trainer"])
        config = {**raw_dict, **config}
        return ExperimentConfig(**config)
