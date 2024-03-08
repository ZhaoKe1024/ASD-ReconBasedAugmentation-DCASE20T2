#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/9/21 17:02
# @Author: ZhaoKe
# @File : loss.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # print("shape of x:", x.shape)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # print("shape of cosine", cosine.shape)
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print("shape of onehot", one_hot.shape)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margim = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, pos, neg):
        part1 = (anchor - pos).pow(2).sum(dim=1)
        part2 = (anchor - neg).pow(2).sum(dim=1)
        return self.relu(part1 - part2 + self.margim).mean()


# only the cel is used here
class D2GLoss(nn.Module):
    '''
        Feature matching loss described in the paper.
    '''

    def __init__(self, cfg):
        super(D2GLoss, self).__init__()
        self.cfg = cfg

    def forward(self, feat_fake, feat_real):
        loss = 0
        norm_loss = {'l2': lambda x, y: F.mse_loss(x, y), 'l1': lambda x, y: F.l1_loss(x, y)}
        stat = {'mu': lambda x: x.mean(dim=0),
                'sigma': lambda x: (x - x.mean(dim=0, keepdim=True)).pow(2).mean(dim=0).sqrt()}

        if 'mu' in self.cfg.keys():
            mu_eff = self.cfg['mu']
            mu_fake, mu_real = stat['mu'](feat_fake), stat['mu'](feat_real)
            norm = norm_loss['l2'](mu_fake, mu_real)
            loss += mu_eff * norm
        if 'sigma' in self.cfg.keys():
            sigma_eff = self.cfg['sigma']
            sigma_fake, sigma_real = stat['sigma'](feat_fake), stat['sigma'](feat_real)
            norm = norm_loss['l2'](sigma_fake, sigma_real)
            loss += sigma_eff * norm
        return loss


if __name__ == '__main__':
    pred = torch.rand(64, 128)
    labe = torch.randint(0, 23, size=(64,))
    arcface = ArcMarginProduct(in_features=128, out_features=23)
    print(arcface(pred, labe).shape)
