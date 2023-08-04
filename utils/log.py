import hydra
import os

from utils.general import colorstr


class Logger:
    def __init__(self, cfg, *logger_types):
        self.cfg = cfg
        self.wandb_logger = None
        self.logging = None

        self.base_path = '~/intestinal_obstruction'
        self._init_logger(logger_types)

    def log(self, data):
        if self.wandb_logger:
            self.wandb_logger.log(data)

    def _init_logger(self, logger_types):
        if len(logger_types) == 0:
            logger_types = ['wandb', 'logging']

        for l_type in logger_types:
            if 'wandb' in l_type.lower():
                try:
                    import wandb

                    wandb.init(project=self.cfg.info.project, entity=self.cfg.info.entity, config=self.cfg,
                               name=f"{self.cfg.name}",
                               settings=wandb.Settings(_disable_stats=True), save_code=True, reinit=True)

                    self.wandb_logger = wandb

                except (ImportError, AssertionError):
                    prefix = colorstr('red', 'Weights & Biases: ')
                    print(f"{prefix}run 'pip install wandb' to track and visualize.")

            if 'logging' in l_type.lower():
                import logging
                self.logging = logging

    def _path(self, x):
        return os.path.join(self.base_path, x)
