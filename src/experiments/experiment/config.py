from dataclasses import dataclass
from enum import Enum
from ..unconditioned.autoencoder.config import ModelConfig as AutoencoderConfig
from ..unconditioned.unet.config import ModelConfig as UnetConfig
from ..unconditioned.dataset.config import DatasetConfig as UnconditionedDatsetConfig
from ..unconditioned.trainer.config import TrainerConfig as UnconditionedTrainerConfig
from src.utils.base_config import BaseConfig
from typing import Dict, Tuple


class ExpType(Enum):
    AUTOENCODER = "autoencoder"
    UNET = "unet"


_exp_type_to_configs: Dict[ExpType, Tuple[BaseConfig, BaseConfig, BaseConfig]] = {
    ExpType.AUTOENCODER: (AutoencoderConfig, UnconditionedDatsetConfig, UnconditionedTrainerConfig),
    ExpType.UNET: (UnetConfig, UnconditionedDatsetConfig, UnconditionedTrainerConfig)
}


@dataclass
class ExperimentConfig(BaseConfig):
    """This class holds configuration details of the experiment"""
    exp_type: ExpType
    model: BaseConfig
    dataset_train: BaseConfig
    dataset_val: BaseConfig
    trainer: BaseConfig
    # If true, copy missing validation dataset keys from train dataset
    val_copy_train: bool = True

    @staticmethod
    def from_dict(raw_dict) -> "ExperimentConfig":
        # TODO Fix this function to accept exp type
        exp_type = ExpType(raw_dict["exp_type"])
        ModelConfig, DatasetConfig, TrainerConfig = _exp_type_to_configs[exp_type]
        model_config = ModelConfig.from_dict(raw_dict["model"])
        dataset_train = DatasetConfig.from_dict(raw_dict["dataset_train"])
        val_copy_train = raw_dict["val_copy_train"]
        if val_copy_train:
            dataset_val = DatasetConfig.from_dict({**raw_dict["dataset_train"], **raw_dict["dataset_val"]})
        else:
            dataset_val = DatasetConfig.from_dict(raw_dict["dataset_val"])
        trainer = TrainerConfig.from_dict(raw_dict["trainer"])
        return ExperimentConfig(exp_type, model_config, dataset_train, dataset_val, trainer, val_copy_train)
