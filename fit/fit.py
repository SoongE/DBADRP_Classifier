import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torchmetrics import Accuracy, MeanMetric, Recall, Specificity, AUROC, ROC, ConfusionMatrix
from torchmetrics.functional import accuracy
from tqdm import tqdm


class Fit:
    def __init__(self, amp, device):
        self.scaler = GradScaler() if amp else None
        self.device = device
        self.local_rank = 0

        self.losses = MeanMetric().to(self.device)
        self.accuracies = Accuracy(task='multiclass', num_classes=3).to(self.device)
        self.sensitivity = Recall(task='multiclass', average='macro', num_classes=3).to(self.device)
        self.specificity = Specificity(task='multiclass', average='macro', num_classes=3).to(self.device)
        self.auroc = AUROC(task='multiclass', num_classes=3).to(self.device)
        self.roc = ROC(task='multiclass', num_classes=3, thresholds=78).to(self.device)
        self.cm = ConfusionMatrix(task='multiclass', num_classes=3).to(self.device)

    def iterate(self, data, model, criterion, is_train=False):
        return torch.tensor(0), torch.tensor(0), torch.tensor(0)

    def train(self, train_loader, model, optimizer, criterion, epoch):
        self._reset_metric()

        if self.local_rank == 0:
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        else:
            pbar = enumerate(train_loader)

        model.train()
        for i, data in pbar:
            optimizer.zero_grad()
            loss, prob, target = self.iterate(data, model, criterion, is_train=True)
            self._backward(loss, optimizer, model)

            self._update_metric(loss, prob, target)
            metrics = self._metrics()

            if self.local_rank == 0:
                pbar.set_description(self._print(metrics, epoch, 'Train'))

        return self._metrics()

    @torch.no_grad()
    def validate(self, val_loader, model, criterion, epoch):
        self._reset_metric()

        if self.local_rank == 0:
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True)
        else:
            pbar = enumerate(val_loader)

        model.eval()
        for i, data in pbar:
            loss, prob, target = self.iterate(data, model, criterion)

            self._update_metric(loss, prob, target)
            metrics = self._metrics()

            if self.local_rank == 0:
                pbar.set_description(self._print(metrics, epoch, 'Val'))

        return self._metrics()

    @torch.no_grad()
    def test(self, test_loader, model, criterion, wrong_item=False):
        self._reset_metric()
        accuracies_list = list()

        if self.local_rank == 0:
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), dynamic_ncols=True)
        else:
            pbar = enumerate(test_loader)

        if isinstance(model, list):
            for m in model:
                m.eval()
        else:
            model.eval()

        for i, data in pbar:
            loss, prob, target = self.iterate(data, model, criterion)

            self._update_metric(loss, prob, target)
            metrics = self._metrics()
            accuracies_list.append(accuracy(prob, target, task='multiclass', num_classes=3))

            if self.local_rank == 0:
                pbar.set_description(self._print(metrics, 0, 'Test'))

        return accuracies_list, self._metrics(), 0

    def _backward(self, loss, optimizer, model):
        if self.scaler:
            self._scaler_backward(loss, optimizer, model)
        else:
            self._default_backward(loss, optimizer, model)

    def _scaler_backward(self, loss, optimizer, model):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 2.0)
        self.scaler.step(optimizer)
        self.scaler.update()

    def _default_backward(self, loss, optimizer, model):
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def _update_metric(self, loss, prob, target):
        self.losses.update(loss.item() / prob.size(0))
        self.accuracies.update(prob, target)
        self.specificity.update(prob, target)
        self.sensitivity.update(prob, target)
        self.auroc.update(prob, target)
        self.roc.update(prob, target)
        self.cm.update(prob, target)

    def _reset_metric(self):
        self.accuracies.reset()
        self.specificity.reset()
        self.sensitivity.reset()
        self.losses.reset()
        self.auroc.reset()
        self.roc.reset()
        self.cm.reset()

    def _metrics(self):
        return {
            'Loss': self.losses.compute(),
            'acc': self.accuracies.compute(),
            'specificity': self.specificity.compute() * 100,
            'sensitivity': self.sensitivity.compute() * 100,
            'auroc': self.auroc.compute(),
            'roc': self.roc.compute(),
            'cm': self.cm.compute(),
        }

    def _print(self, metrics, epoch, mode):
        log = f'[{mode}#{epoch:>3}] '
        for k, v in metrics.items():
            if k in ['roc', 'cm']:
                continue
            log += f'{k}:{v:.6f} | '
        return log[:-3]
