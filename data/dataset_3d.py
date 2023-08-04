import json
import os
import warnings

import SimpleITK as itk
import torch
import torchio as tio

itk.ProcessObject.SetGlobalWarningDisplay(False)
warnings.filterwarnings('ignore')


def set_transform(size, depth, resample, mode):
    preprocess = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(resample),
        tio.CropOrPad((size, size, depth)),
    ])

    normalization = tio.Compose([
        tio.ZNormalization(tio.ZNormalization.mean),
    ])
    augment = tio.Compose([
        tio.RandomFlip(),
        tio.RandomGamma(p=0.5),
        tio.RandomNoise(p=0.5),
        tio.RandomMotion(p=0.1),
        tio.RandomBiasField(p=0.25),
        tio.OneOf({
            tio.RandomAffine(): 0.8,
            tio.RandomElasticDeformation(): 0.2,
        }),
    ])

    if mode == 'train':
        transform = tio.Compose([preprocess, augment, normalization])
    elif mode == 'origin':
        transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(resample),
            tio.CropOrPad((size, size, depth)),
        ])
    else:
        transform = tio.Compose([preprocess, normalization])

    return transform


class Dataset3D:
    def __init__(self, cfg, mode, finetune=False):
        super().__init__()
        self.cfg = cfg
        self.root = self.cfg.root
        self.mode = mode
        self.finetune = finetune

        self.subjects = self.load_data(mode)
        self.transform = set_transform(self.cfg.size, self.cfg.depth, self.cfg.resample, self.mode)

    def load_data(self, mode):
        file_name = os.path.join(self.root, 'splits', self.cfg.json_file + '.json')
        with open(file_name, 'r') as f:
            self.data = json.load(f)

        subjects = list()
        for image_path, l in zip(self.data[mode], self.data[f'{mode}_label']):
            if not self.finetune:
                l = 1 if l == 2 else l
            subject = tio.Subject(
                ct=tio.ScalarImage(image_path),
                label=torch.tensor(l),
                image_path=image_path
            )
            subjects.append(subject)
        return subjects

    def get_dataset(self):
        return tio.SubjectsDataset(self.subjects, transform=self.transform)


class KFoldDataset3D:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root = self.cfg.root

        self.data = None
        self.load_data()

    def load_data(self):
        file_name = os.path.join(self.root, 'splits', self.cfg.json_file + '.json')
        with open(file_name, 'r') as f:
            self.data = json.load(f)

    def set_subject(self, k, mode):
        subjects = list()

        for image_path, l in zip(self.data[k][mode], self.data[k][f'{mode}_label']):
            subject = tio.Subject(
                ct=tio.ScalarImage(image_path),
                label=torch.tensor(l),
            )
            subjects.append(subject)

        return subjects

    def get_dataset(self, k, mode):
        subject = self.set_subject(k, mode)
        transform = set_transform(self.cfg.size, self.cfg.depth, self.cfg.resample, mode)
        return tio.SubjectsDataset(subject, transform=transform)


class TestDataset3D:
    def __init__(self, cfg, mode, k):
        super().__init__()
        self.cfg = cfg
        self.root = self.cfg.root
        self.mode = mode
        self.k = k

        self.abnormal = list()  # 23
        self.weak_abnormal = list()  # 20
        self.normal = list()  # 35
        self.load_data(mode)

        self.resize_transform = set_transform(self.cfg.size, self.cfg.depth, self.cfg.resample, 'origin')
        self.input_transform = set_transform(self.cfg.size, self.cfg.depth, self.cfg.resample, 'test')

    def load_data(self, mode):
        file_name = os.path.join(self.root, 'splits', self.cfg.json_file + '.json')
        with open(file_name, 'r') as f:
            self.data = json.load(f)[self.k]

        for image_path, l in zip(self.data[mode], self.data[f'{mode}_label']):
            subject = tio.Subject(
                ct=tio.ScalarImage(image_path),
                label=torch.tensor(l),
                image_path=image_path
            )
            if l == 0:
                self.normal.append(subject)
            if l == 1:
                self.abnormal.append(subject)
            if l == 2:
                self.weak_abnormal.append(subject)

    def get_images(self, idx, dtype='normal'):
        if dtype == 'normal':
            subject = self.normal[idx]
        elif dtype == 'high':
            subject = self.abnormal[idx]
        elif dtype == 'low':
            subject = self.weak_abnormal[idx]
        else:
            raise ValueError

        resize_image = self.resize_transform(subject).ct.data
        transformed_image = self.input_transform(subject).ct.data
        label = subject.label.data

        return resize_image, transformed_image, label
