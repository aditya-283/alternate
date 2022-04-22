import torch
import torch.nn as nn
from models import *


def factorize(linear_layer, rank_ratio):
    """Converts torch.nn.Linear into LinearLR"""
    weight = linear_layer.weight
    rank = min(weight.size()[0], weight.size()[1])
    sliced_rank = int(rank * rank_ratio)

    # factorize original weights
    u, s, v = torch.svd(weight)
    u_weight = (u * torch.sqrt(s))[:, 0:sliced_rank]
    v_weight = (torch.sqrt(s) * v)[:, 0:sliced_rank]
    res_weight = weight - u_weight.matmul(v_weight.t())

    # extract arguments
    in_features, out_features = linear_layer.in_features, linear_layer.out_features
    device, dtype = weight.device, weight.dtype
    bias = linear_layer.bias is not None

    lowrank_layer = LinearLR(in_features, out_features, rank_ratio, bias, device, dtype)

    # initialize lowrank layer weights with factorized weights
    with torch.no_grad():
        lowrank_layer.u.weight.copy_(u_weight)
        lowrank_layer.v.weight.copy_(v_weight.t())
        lowrank_layer.res.weight.copy_(res_weight)

        if bias is not None:
            lowrank_layer.u.bias.copy_(linear_layer.bias)

    return lowrank_layer


class LinearLR(nn.Module):
    """[u * v + res] version of torch.nn.Linear"""

    def __init__(
        self,
        in_features,
        out_features,
        rank_ratio=0.25,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        sliced_rank = int(min(in_features, out_features) * rank_ratio)
        self.u = nn.Linear(
            in_features, sliced_rank, bias=False, device=device, dtype=dtype
        )

        self.v = nn.Linear(
            sliced_rank,
            out_features,
            bias=bias,  # original bias is stored as the bias of v
            device=device,
            dtype=dtype,
        )

        self.res = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

    def freeze(self):
        for param in self.res.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.res.parameters():
            param.requires_grad = True

    def forward(self, input):
        return self.v(self.u(input)) + self.res(input)


class ConvLR(nn.Module):
    """[u * v + res] version of torch.nn.ConvLR"""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        rank_ratio=0.25,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        sliced_rank = int(min(in_planes, out_planes) * rank_ratio)
        self.u = nn.Conv2d(
            in_channels=in_planes,
            out_channels=sliced_rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype
        )

        self.v = nn.Conv2d(
            in_channels=sliced_rank,
            out_channels=out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            device=device,
            dtype=dtype
        )

        self.res = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype
        )


    def forward(self, input):
        return self.v(self.u(input)) + self.res(input)


def factorizeLayer(layer, rank_ratio):
    in_channels, out_channels =  layer.in_channels, layer.out_channels
    kernel, stride, padding = layer.kernel_size[0], layer.stride[0], layer.padding
    weight = layer.weight
    param_reshaped = weight.view(weight.size()[0], -1)
    rank = min(param_reshaped.size())
    u, s, v = torch.svd(param_reshaped)
    sliced_rank = int(rank * rank_ratio)
    u_weight = (u * torch.sqrt(s))[:, 0:sliced_rank]
    v_weight = (torch.sqrt(s) * v)[:, 0:sliced_rank]
    approx = torch.mm(u_weight, v_weight.t())
    res = param_reshaped - approx
    u_weight_shape, v_weight_shape = (u_weight.size(), v_weight.size())
    model_weight_v = u_weight.view(u_weight_shape[0], u_weight_shape[1], 1, 1)
    model_weight_u = v_weight.t().view(v_weight_shape[1], weight.size()[1], weight.size()[2], weight.size()[3])
    model_res = res.view(weight.size())

    lowrank = ConvLR(in_channels, out_channels, kernel, stride, padding, rank_ratio, layer.bias is not None, layer.weight.device, layer.weight.dtype)
    with torch.no_grad():
        lowrank.u.weight.copy_(model_weight_u)
        lowrank.v.weight.copy_(model_weight_v)
        lowrank.res.weight.copy_(model_res)
    
    return lowrank


import functools
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def factorizeModel(model, rank_ratio): # in-place
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and min(module.weight.size()[:2]) > 4 and module.stride[0] == 1:
            factorizedLayer = factorizeLayer(rgetattr(model, name), rank_ratio)
            rsetattr(model, name, factorizedLayer)

def unfreezeResidual(model):
    for name, param in model.named_parameters():
        if "res" in name:
            # print(name)
            param.requires_grad = True

def freezeResidual(model):
    for name, param in model.named_parameters():
        if "res" in name:
            # print(name)
            param.requires_grad = False

resnet = ResNet18()
print(resnet)
factorizeModel(resnet, 0.25)
print(resnet)
