import torch
import torch.nn as nn
from torch.nn import functional as F


def conv(nin, nout, kernel_size=3, stride=1, padding=1, bias=True, layer=nn.Conv2d, norm_layer=None, num_groups=None,
         activ=nn.ReLU(inplace=True), maxpool=None, **kwargs):
    layers = []
    layers.append(
        layer(nin, nout, kernel_size, stride=stride, padding=0 if kernel_size == 1 else padding, bias=bias, **kwargs))
    if norm_layer is not None:
        layers.append(norm(norm_layer, nout, num_groups))
    if activ is not None:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
            layers.append(activ(num_parameters=1))
        else:
            layers.append(activ)
    if maxpool is not None:
        if isinstance(maxpool, (list, tuple)):
            layers.append(nn.MaxPool2d(*maxpool))
        else:
            layers.append(nn.MaxPool2d(3, 2, 1))
    return nn.Sequential(*layers)


def fc(nin, nout, bias=True, layer=nn.Linear, BN=False, activ=nn.ReLU(inplace=True)):
    layers = []
    layers.append(layer(nin, nout, bias=bias))
    if BN:
        layers.append(nn.BatchNorm1d(nout))
    if activ is not None:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()')
            layers.append(activ(num_parameters=1))
        else:
            layers.append(activ)
    return nn.Sequential(*layers)


def norm(norm_layer, num_channels=None, num_groups=None):
    if num_groups is None:
        return norm_layer(num_channels)
    else:
        return norm_layer(num_groups, num_channels)