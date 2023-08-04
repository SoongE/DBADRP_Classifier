import torch
import torchio as tio
from torch.nn.functional import softmax

from fit.fit import Fit


class BaseFit3D(Fit):
    def __init__(self, device, amp=False):
        super().__init__(amp, device)

    def iterate(self, data, model, criterion, is_train=False):
        x, y = data['ct'][tio.DATA].to(self.device), data['label'].to(self.device)
        x = x.permute(0, 1, 4, 2, 3)

        prob = model(x)
        loss = criterion(prob, y)

        return loss, prob, y


class BaseFit3DPlus(Fit):
    def __init__(self, device, amp=False):
        super().__init__(amp, device)
        self.dba = False

    def iterate(self, data, model, criterion, is_train=False):
        x, y = data['ct'][tio.DATA].to(self.device), data['label'].to(self.device)
        x = x.permute(0, 1, 4, 2, 3)

        if self.dba:
            prob, feature = model(x, last_feature=True)
            ab_idx = (y != 0).nonzero(as_tuple=True)[0]
            ab_prob = model.abnormal_fc(feature[ab_idx])
            loss = criterion(prob, y) + criterion(ab_prob, y[ab_idx] - 1) / 2

        else:
            prob = model(x)
            loss = criterion(prob, y)

        return loss, prob, y


class BaseFit3DPlusEnsemble(Fit):
    def __init__(self, device, amp=False):
        super().__init__(amp, device)
        self.dba = False

    def iterate(self, data, models, criterion, is_train=False):
        x, y = data['ct'][tio.DATA].to(self.device), data['label'].to(self.device)
        x = x.permute(0, 1, 4, 2, 3)

        probs = list()
        losses = list()
        for model in models:
            prob = model(x)
            loss = criterion(prob, y)
            probs.append(softmax(prob, dim=-1))
            losses.append(loss)

        prob = torch.stack(probs, dim=1)
        prob = prob.mean(dim=1)
        loss = torch.stack(losses, dim=0)
        loss = loss.mean(dim=0)

        return loss, prob, y
