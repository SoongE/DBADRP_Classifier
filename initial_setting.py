import numpy as np
import torch
from torch import nn

from fit.basefit import BaseFit3DPlus
from models.densenet import DenseNet121
from models.efficientnet import EfficientNetBN
from models.module import SEModule
from models.resnet import resnet50, resnext50, wideresnet50


def get_optimizer(model, lr, weight_decay, optimizer='Adam'):
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    return optimizer


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True


def implement_dba(model, backbone):
    if 'res' in backbone:
        scale = 4 if 'drp' in backbone else 1
        feature = 2048
    elif 'efficientnet' in backbone:
        scale = 3 if 'drp' in backbone else 1
        feature = 1280
    elif 'densenet' in backbone:
        scale = 3 if 'drp' in backbone else 1
        feature = 1024
    else:
        raise ValueError(f'{backbone} feature is not defined.')

    ab_fc_layer = ('abnormal_fc', nn.Linear(feature * scale, 2))
    model.add_module(*ab_fc_layer)
    return model


def implement_drp(model, backbone, n_classes):
    backbone = backbone.split('-')[0]

    if 'res' in backbone:
        scale = 4
        channels = 2048
        model.avgpool = nn.Sequential(
            SEModule(channels, reduction=4),
            nn.AdaptiveAvgPool3d((scale, 1, 1)),
        )
        model.fc = nn.Linear(channels * scale, n_classes)
    elif 'efficientnet' in backbone:
        scale = 3
        channels = 1280
        model._avg_pooling = nn.Sequential(
            SEModule(channels, reduction=4),
            nn.AdaptiveAvgPool3d((scale, 1, 1)),
        )
        model._fc = nn.Linear(channels * scale, n_classes)
    elif 'densenet' in backbone:
        scale = 3
        channels = 1024
        model.flatten.pool = nn.Sequential(
            SEModule(channels, reduction=4),
            nn.AdaptiveAvgPool3d((scale, 1, 1)),
        )
        model.last_linear = nn.Linear(channels * scale, n_classes)
    else:
        raise ValueError(f'{backbone} feature is not defined.')
    return model


def get_instance(cfg, device):
    model = _get_backbone(cfg)
    model = implement_drp(model, cfg.backbone, cfg.dataset.num_class) if 'drp' in cfg.backbone else model
    model = implement_dba(model, cfg.backbone) if 'dba' in cfg.backbone else model
    model.initial_weights()

    criterion, run = _get_instance(cfg, device)
    run.dba = True if 'dba' in cfg.backbone else False

    if torch.cuda.is_available():
        model.to(device)
        criterion.to(device)

    return model, criterion, run


def _get_backbone(cfg):
    backbone = cfg.backbone.split('-')[0]

    if backbone == 'resnet':
        return resnet50(n_input_channels=1, spatial_dims=3, num_classes=cfg.dataset.num_class)
    elif backbone == 'resnext':
        return resnext50(n_input_channels=1, spatial_dims=3, num_classes=cfg.dataset.num_class)
    elif backbone == 'wideresnet':
        return wideresnet50(n_input_channels=1, spatial_dims=3, num_classes=cfg.dataset.num_class)
    elif backbone == 'efficientnet':
        return EfficientNetBN("efficientnet-b0", in_channels=1, spatial_dims=3, num_classes=cfg.dataset.num_class)
    elif backbone == 'densenet':
        return DenseNet121(in_channels=1, spatial_dims=3, num_classes=cfg.dataset.num_class)
    else:
        raise ValueError(f"{backbone} is not defined")


def _get_instance(cfg, device):
    if 'baseline3d' == cfg.model:
        run = BaseFit3DPlus(device)
        criterion = torch.nn.CrossEntropyLoss()
        return criterion, run
    else:
        raise AttributeError(f"There is no {cfg.model} model")


if __name__ == '__main__':
    model = resnet50(n_input_channels=1, spatial_dims=3, num_classes=3)
    input = torch.rand(2, 1, 112, 224, 224)
    print(model(input).shape)
