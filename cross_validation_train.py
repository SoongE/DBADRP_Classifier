import os
from datetime import timedelta

import hydra
import torch
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from torch.distributed import init_process_group
from wandb import AlertLevel

from data.loader import get_kfold_dataloader3d
from initial_setting import init_seed, get_instance
from main import train
from utils.log import Logger


def main_worker(local_rank, ngpus_per_node, cfg):
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    if cfg.distributed:
        init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=local_rank)
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        cfg.local_rank = local_rank
        cfg.world_size = torch.distributed.get_world_size()

    logger = None

    for k in range(1, 5):
        if local_rank == 0:
            logger = Logger(cfg)

        init_seed(cfg.train.seed)

        train_loader, val_loader = get_kfold_dataloader3d(cfg, k, cfg.dataset.trainset_name, cfg.dataset.validset_name)
        model, criterion, run = get_instance(cfg, device)

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=cfg.train))
        scheduler, num_epochs = create_scheduler(cfg.train, optimizer)

        if cfg.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            run.local_rank = local_rank

        model, best_accuracy = train(cfg, run, train_loader, val_loader, model, optimizer, criterion, scheduler, logger,
                                     k)

        logger.wandb_logger.alert(
            title=f'({cfg.info.project}) Run finished',
            text=f'Best:{best_accuracy:.1f}',
            level=AlertLevel.INFO,
            wait_duration=timedelta(minutes=10)
        )


@hydra.main(config_path='configs', config_name="config")
def main(cfg) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in cfg.gpus)
    ngpus_per_node = torch.cuda.device_count()

    cfg.distributed = ngpus_per_node > 1
    cfg.name = cfg.backbone if cfg.name is None else cfg.name

    cfg.dataset.num_class = 3

    if cfg.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(0, ngpus_per_node, cfg)


if __name__ == "__main__":
    main()
