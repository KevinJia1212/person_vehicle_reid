#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class TripletSemihardLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, margin=0):
        super(TripletSemihardLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        y_true = target.int().unsqueeze(-1)
        same_id = torch.eq(y_true, y_true.t()).type_as(input)

        pos_mask = same_id
        neg_mask = 1 - same_id

        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor - 1e6 * (1 - mask)
            _max, _idx = torch.max(input_tensor, dim=axis, keepdim=keepdims)
            return _max, _idx

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1e6 * (1 - mask)
            _min, _idx = torch.min(input_tensor, dim=axis, keepdim=keepdims)
            return _min, _idx

        # output[i, j] = || feature[i, :] - feature[j, :] ||_2
        dist_squared = torch.sum(input ** 2, dim=1, keepdim=True) + \
                       torch.sum(input.t() ** 2, dim=0, keepdim=True) - \
                       2.0 * torch.matmul(input, input.t())
        dist = dist_squared.clamp(min=1e-16).sqrt()

        pos_max, pos_idx = _mask_max(dist, pos_mask, axis=-1)
        neg_min, neg_idx = _mask_min(dist, neg_mask, axis=-1)

        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        y = torch.ones(same_id.size()[0]).cuda()
        triloss = F.margin_ranking_loss(neg_min.float(),
                                     pos_max.float(),
                                     y,
                                     self.margin)
        prec = (neg_min.data > pos_max.data).sum() * 1. / same_id.size()[0]
        return triloss, prec

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin)

#     def forward(self, inputs, targets):
#         n = inputs.size(0)
#         # Compute pairwise distance, replace by the official when merged
#         dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#         dist = dist + dist.t()
#         dist.addmm_(1, -2, inputs, inputs.t())
#         dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max())
#             dist_an.append(dist[i][mask[i] == 0].min())
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
#         # Compute ranking hinge loss
#         y = dist_an.data.new()
#         y.resize_as_(dist_an.data)
#         y.fill_(1)
#         y = Variable(y)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
#         return loss, prec
