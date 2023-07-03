import yaml
import sys
from .config import ExperimentConfig, ExpType

if __name__ == '__main__':
    cfg_path = "./config/example.yml"
    if len(sys.argv) == 2:
        cfg_path = sys.argv[-1]
    with open(cfg_path, "r") as fp:
        raw_dict = yaml.safe_load(fp)
    exp_config = ExperimentConfig.from_dict(raw_dict["experiment"])
    if exp_config.exp_type in (ExpType.AUTOENCODER, ExpType.UNET):
        from ..unconditioned.trainer.trainer import Trainer
    else:
        pass
    trainer = Trainer(exp_config)
    trainer.train()
