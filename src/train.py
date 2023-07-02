import yaml
import sys
from enum import Enum
from rich import print


class ExperimentType(Enum):
    AUTOENCODER = "autoencoder"


if __name__ == "__main__":
    config_path = "./config/example.yml"
    if len(sys.argv) == 2:
        config_path = sys.argv[-1]

    with open(config_path, "r") as fp:
        raw_dict = yaml.safe_load(fp)

    experiment_type = ExperimentType(raw_dict["experiment_type"].strip().lower())
    if experiment_type == ExperimentType.AUTOENCODER:
        from src.experiments.autoencoder.config import ExperimentConfig
        from src.experiments.autoencoder.trainer import Trainer

    cfg = ExperimentConfig.from_dict(raw_dict["experiment_config"])
    print("Config:", cfg)
    trainer = Trainer(cfg)
    trainer.train()
