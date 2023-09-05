#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 18/05/2023 8:07 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""

import torch
import torch.nn.functional as F


def l1_loss(inputs, targets):
    """ L1 Loss without reduce flag.

    Args:
        inputs (FloatTensor): Input tensor
        targets (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(inputs - targets))


def l2_loss(inputs, targets, average_flag=True):
    """ L2 Loss without reduce flag.

    Args:
        inputs (FloatTensor): Input tensor
        targets (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """

    if average_flag:
        return torch.mean(torch.pow((inputs-targets), 2))
    else:
        return torch.pow((inputs-targets), 2)

def sce_loss(x, y, alpha=1, reduction=True):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    if reduction:
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
    else:
        loss = (1 - (x * y))

    return loss
