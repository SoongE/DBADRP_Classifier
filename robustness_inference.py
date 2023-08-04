import json
import os

import pandas as pd
import torch
import torchio as tio
from hydra import initialize, compose
from torch.utils.data import DataLoader

from data.dataset_3d import set_transform
from initial_setting import get_instance


class TestDataset3D:
    def __init__(self, cfg, distort=False):
        super().__init__()
        self.cfg = cfg.dataset
        self.root = '/data/intestinal_obstruction'
        self.distort = distort
        if distort:
            self.distort_root = f'distort/{distort}/'

        self.data = None
        self.load_data()

    def load_data(self):
        file_name = os.path.join(self.root, 'splits', self.cfg.json_file + '.json')
        with open(file_name, 'r') as f:
            self.data = json.load(f)

    def set_subject(self, k, mode):
        subjects = list()

        for image_path, l in zip(self.data[k][mode], self.data[k][f'{mode}_label']):
            if self.distort:
                image_path = os.path.join(self.root, self.distort_root, image_path.split('/', 3)[-1]) + '.nii.gz'
            subject = tio.Subject(
                ct=tio.ScalarImage(image_path),
                label=torch.tensor(l),
            )
            subjects.append(subject)

        return subjects

    def get_dataset(self, k):
        subject = self.set_subject(k, 'valid')
        transform = set_transform(self.cfg.size, self.cfg.depth, self.cfg.resample, 'test')
        return tio.SubjectsDataset(subject, transform=transform)


def main():
    with initialize('configs'):
        cfg = compose('config', overrides=['train.batch_size=8'])

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in cfg.gpus)
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')

    backbones = ['resnet', 'wideresnet', 'resnext', 'densenet', 'efficientnet']
    methods = ['-dbadrp', '']
    distorts = ['affine', 'gamma+', 'gamma']
    res = pd.DataFrame()

    for k in range(5):
        for backbone in backbones:
            for m in methods:
                torch.cuda.empty_cache()

                weight_files = f'{backbone}{m}{k}k_best.pt'
                cfg.backbone = f'{backbone}{m}'
                weight_files = os.path.join('weights/', weight_files)

                model, criterion, run = get_instance(cfg, device)
                print(f'Load {weight_files}')
                model.load_state_dict(torch.load(weight_files)['state_dict'])
                model.eval()

                for d in distorts:
                    loader = DataLoader(TestDataset3D(cfg, d).get_dataset(k), batch_size=cfg.train.batch_size)
                    metrics = run.validate(loader, model, criterion, 0)
                    metrics.pop('cm')
                    metrics.pop('roc')
                    output = pd.DataFrame([metrics]).astype("float")
                    output['model'] = backbone
                    output['methods'] = m
                    output['corrupt'] = d

                    res = pd.concat([res, output], ignore_index=True)

                    res_print = f'{backbone}{m}/{d}: {metrics}'
                    print(res_print + '\n')
    res.sort_values(by=['corrupt', 'model', 'methods'])
    res.to_csv('~/intestinal_obstruction/inference.csv')
    print(res)


if __name__ == '__main__':
    main()
