import yaml
import sys
from .config import ExperimentConfig, ExpType
from ..unconditioned.trainer.trainer import Trainer as UnconditionedTrainer
from ..conditioned.trainer.trainer import Trainer as ConditionedTrainer
from rich import print


def main(caller=None):
    cfg_path = "./config/example.yml"
    if len(sys.argv) == 2:
        cfg_path = sys.argv[-1]
    with open(cfg_path, "r") as fp:
        raw_dict = yaml.safe_load(fp)
    exp_config = ExperimentConfig.from_dict(raw_dict["experiment"])
    if exp_config.exp_type in (ExpType.AUTOENCODER, ExpType.UNET):
        trainer = UnconditionedTrainer(exp_config)        
    else:
        trainer = ConditionedTrainer(exp_config)
    print(exp_config)
    trainer.train('__main__')


if __name__ == '__main__':
    main(__name__)
