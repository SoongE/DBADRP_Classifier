import os
from datetime import timedelta

import hydra
import torch
from torch.distributed import init_process_group
from wandb import AlertLevel

from data.loader import get_dataloader3d
from initial_setting import init_seed, get_instance, get_optimizer
from utils.calculate import margin_of_error
from utils.general import save_model
from utils.log import Logger

torch.multiprocessing.set_sharing_strategy('file_system')


def add_type_in_log(metrics, dtype):
    new_dict = dict()
    for k, v in metrics.items():
        new_dict[f'{dtype}_{k}'] = v
    return new_dict


def train(cfg, run, train_loader, val_loader, model, optimizer, criterion, scheduler, logger, k=None):
    best_acc1 = 0
    best_model = model

    distributed = cfg.distributed
    name = 'run' if cfg.name is None else cfg.name
    name = name if k is None else name + f'{k}k'
    cfg = cfg.train

    for epoch in range(cfg.epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = run.train(train_loader, model, optimizer, criterion, epoch)
        val_metrics = run.validate(val_loader, model, criterion, epoch)

        if val_metrics['auroc'] >= best_acc1:
            is_best = True
            best_acc1 = val_metrics['auroc']
        else:
            is_best = False

        val_metrics['best_auroc'] = best_acc1
        log = {**add_type_in_log(train_metrics, 'train'), **add_type_in_log(val_metrics, 'val')}

        if run.local_rank == 0:
            logger.log(log)
            logger.wandb_logger.run.summary["best_auroc"] = best_acc1

        if is_best:
            best_model = model
            save_model(best_model, name + '_best', best_acc1)

        scheduler.step(epoch + 1)

    save_model(best_model, name + '_final', best_acc1)

    return best_model, best_acc1


def test(run, test_loader, model, criterion, logger):
    test_loss, test_acc, _ = run.test(test_loader, model, criterion)
    value, moe = margin_of_error(test_acc)

    log = {'test_acc': value}
    logger.log(log)


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
    if local_rank == 0:
        logger = Logger(cfg)

    init_seed(cfg.train.seed)

    train_loader, val_loader = get_dataloader3d(cfg, False, cfg.dataset.trainset_name, cfg.dataset.validset_name)
    model, criterion, run = get_instance(cfg, device)

    optimizer = get_optimizer(model, cfg.train.lr, cfg.train.weight_decay, optimizer=cfg.train.optimizer)
    eta_min = cfg.train.lr * (cfg.train.lr_decay ** 3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs, eta_min, -1)

    if cfg.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        run.local_rank = local_rank

    model, best_accuracy = train(cfg, run, train_loader, val_loader, model, optimizer, criterion, scheduler, logger)

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

    if cfg.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(0, ngpus_per_node, cfg)


if __name__ == "__main__":
    main()
