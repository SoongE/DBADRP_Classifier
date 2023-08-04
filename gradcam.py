import os

import matplotlib.pyplot as plt
import torch
from hydra import initialize, compose
from monai.visualize import GradCAM
from torch.nn.functional import softmax
from tqdm import tqdm

from data.dataset_3d import TestDataset3D
from initial_setting import get_instance


def run(data, model, device, target_layer, label):
    torch.cuda.empty_cache()

    origin, transformed, label = data
    transformed = transformed.type(torch.FloatTensor)
    transformed = transformed.unsqueeze(0).permute(0, 1, 4, 2, 3)
    origin = origin.unsqueeze(0).permute(0, 1, 4, 2, 3)
    origin, transformed, label = origin.to(device), transformed.to(device), label.to(device)
    cam = GradCAM(nn_module=model, target_layers=target_layer)(x=transformed, class_idx=label)

    origin = origin.detach().cpu()
    cam = cam.detach().cpu()

    prob = model(transformed)
    pred = prob.argmax(dim=1)[0]
    prob = softmax(prob, dim=1)[0]

    return origin, cam, label, pred, prob


def show_cam(origin, cam, label, pred, prob, index=30, show=False):
    plt.axis('off')  # x,y축 모두 없애기
    data = cam[0, :, index, :, :].permute(1, 2, 0)
    origin = origin[0, :, index, :, :].permute(1, 2, 0)

    data = data - torch.min(data)
    data = data / torch.max(data)
    data = -data
    data[data < -0.6] = data[data < -0.6] * 1.5
    data = torch.nan_to_num(data, 0.0001)

    result = label == pred

    plt.title(f'{prob[label.item()]:.2f} {result}', color='g' if result else 'r', fontsize=18)
    plt.imshow(origin, cmap='gray')
    plt.imshow(data, alpha=0.5, cmap='jet')
    if show:
        plt.show()


def load_model(cfg, device, k):
    weight_files = f'weights/{cfg.backbone}{k}k_final.pt'
    model, criterion, _ = get_instance(cfg, device)
    model.to(device)
    print(f'Load {weight_files}')
    model.load_state_dict(torch.load(weight_files)['state_dict'])
    model.eval()
    return model


def make_ttt_cam(model, normal, high, low, device, interval, tl, save):
    intervals = range(30, 80, interval)
    fig = plt.figure(figsize=(12, 12))
    column = 5
    row = int(len(intervals) * 3 / column)
    length = column * row // 3

    origin, cam, label, pred, prob = run(normal, model, device, tl, 0)
    for i, index in enumerate(intervals):
        fig.add_subplot(row, column, i + 1)
        fig.tight_layout()
        show_cam(origin, cam, label, pred, prob, index=index, show=False)

    origin, cam, label, pred, prob = run(high, model, device, tl, 1)
    for i, index in enumerate(intervals):
        fig.add_subplot(row, column, i + 1 + length)
        fig.tight_layout()
        show_cam(origin, cam, label, pred, prob, index=index, show=False)

    origin, cam, label, pred, prob = run(low, model, device, tl, 2)
    for i, index in enumerate(intervals):
        fig.add_subplot(row, column, i + 1 + length * 2)
        fig.tight_layout()
        show_cam(origin, cam, label, pred, prob, index=index, show=False)

    if save != '':
        plt.tight_layout()
        plt.savefig(save + '.svg', format='svg', dpi=500)
    plt.show()


def make_group_cam(model, normal, high, low, n_data, device, index, tl, save):
    fig = plt.figure(figsize=(10, 5))

    column = 20
    row = int(n_data * 3 / column)
    length = column * row // 3
    for i in tqdm(range(n_data), desc='extract normals'):
        origin, cam, label, pred, prob = run(normal[i], model, device, tl, 0)
        fig.add_subplot(row, column, i + 1)
        fig.tight_layout()
        show_cam(origin, cam, label, pred, prob, index=index, show=False)

    for i in tqdm(range(n_data), desc='extract high'):
        origin, cam, label, pred, prob = run(high[i], model, device, tl, 1)
        fig.add_subplot(row, column, i + 1 + length)
        fig.tight_layout()
        show_cam(origin, cam, label, pred, prob, index=index, show=False)

    for i in tqdm(range(n_data), desc='extract low'):
        origin, cam, label, pred, prob = run(low[i], model, device, tl, 2)
        fig.add_subplot(row, column, i + 1 + length * 2)
        fig.tight_layout()
        show_cam(origin, cam, label, pred, prob, index=index, show=False)

    if save != '':
        plt.tight_layout()
        plt.savefig(save + '.svg', format="svg", dpi=500)
    plt.show()


def main():
    with initialize('configs'):
        cfg = compose('config', overrides=['train.batch_size=8', 'dataset=asbo3-k'])

    k = 0
    n_data = 20
    index = 56
    interval = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in cfg.gpus)
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = TestDataset3D(cfg.dataset, cfg.dataset.trainset_name, k)

    if n_data == 1:
        normal = dataset.get_images(0, 'normal')
        high = dataset.get_images(0, 'high')
        low = dataset.get_images(0, 'low')
        print("CT Loaded")
    else:
        normal = list()
        high = list()
        low = list()
        for i in range(n_data):
            normal.append(dataset.get_images(i, 'normal'))
            high.append(dataset.get_images(i, 'high'))
            low.append(dataset.get_images(i, 'low'))
            print(f'{i}: {dataset.abnormal[i].image_path} / {dataset.weak_abnormal[i].image_path}')

    backbones = ['resnet', 'wideresnet', 'resnext', 'densenet', 'efficientnet']
    target_layer = ['layer4', 'layer4', 'layer4', 'features.denseblock4', '_bn1']
    methods = ['-dbadrp'] * 5

    for backbone, tl in zip(backbones, target_layer):
        for m in methods:
            cfg.backbone = f'{backbone}{m}'
            torch.cuda.empty_cache()

            model = load_model(cfg, device, k)
            if n_data == 1:
                make_ttt_cam(model, normal, high, low, device, interval, tl, save=f'cam_out/{cfg.backbone}_OnlyOneCAM')
            else:
                make_group_cam(model, normal, high, low, n_data, device, index, tl,
                               save=f'cam_out/{cfg.backbone}_groupCAM')


if __name__ == '__main__':
    main()
