from hydra import compose, initialize
from torch.utils.data import DataLoader
from torchvision.datasets.samplers import DistributedSampler

from data.dataset_3d import Dataset3D, KFoldDataset3D


def get_dataloader3d(cfg, finetune, *modes):
    res = list()
    for mode in modes:
        dataset = Dataset3D(cfg.dataset, mode, finetune=finetune).get_dataset()

        if cfg.distributed:
            sampler = DistributedSampler(dataset)
            shuffle = False
            drop_last = True if 'train' in mode else False
        else:
            sampler = None
            shuffle = True
            drop_last = False

        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=shuffle, drop_last=drop_last,
                            sampler=sampler, num_workers=cfg.train.num_workers, pin_memory=True)

        res.append(loader)

    if len(res) == 1:
        return res[0]
    else:
        return res


def get_kfold_dataloader3d(cfg, k, *modes):
    res = list()
    kfold = KFoldDataset3D(cfg.dataset)
    for mode in modes:
        dataset = kfold.get_dataset(k, mode)

        if cfg.distributed:
            sampler = DistributedSampler(dataset)
            shuffle = False
            drop_last = True if 'train' in mode else False
        else:
            sampler = None
            shuffle = True
            drop_last = False

        loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=shuffle, drop_last=drop_last,
                            sampler=sampler, num_workers=cfg.train.num_workers, pin_memory=True)

        res.append(loader)

    if len(res) == 1:
        return res[0]
    else:
        return res


if __name__ == '__main__':
    with initialize(config_path='../configs/'):
        cfg = compose(config_name='config.yaml')
    cfg.train.batch_size = 2
    train_loader, val_loader, test_loader = get_dataloader3d(cfg, cfg.dataset.trainset_name, cfg.dataset.validset_name,
                                                             cfg.dataset.testset_name)

    next(iter(train_loader))
    next(iter(val_loader))
    next(iter(test_loader))
