# Copyright (c) 2021 Dai HBG


"""
Implementation of basic nf methods.

log
2021-12-09
-- init
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from time import time


class PlanarNet(nn.Module):  # planar flow
    """
    f(z) = z + u * h(w * z + b)
    """

    def __init__(self, input_dim, u=None, w=None, b=None, activation_func='tanh', device='cpu'):
        """
        :param input_dim: dimension of z, a tuple is needed
        :param u:
        :param w:
        :param b:
        :param activation_func: activation_function
        :param device:
        """
        super(PlanarNet, self).__init__()

        self.device = device
        self.input_dim = input_dim

        # initialize parameters
        if u is None:
            self.u = nn.Parameter(torch.empty(input_dim)[None])
            nn.init.uniform_(self.u, -1, 1)
        else:
            self.u = u
        if w is None:
            self.w = nn.Parameter(torch.empty(input_dim)[None])
            nn.init.uniform_(self.w, -1, 1)
        else:
            self.w = w
        if b is None:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = b

        if activation_func == 'tanh':
            self.h = torch.tanh
        elif activation_func == 'leaky_relu':
            self.h = nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError('Fatal activation function. Please use tanh or leakyrelu')

        self.activation_func = activation_func

    def forward(self, z):
        """
        :param z: current samples, with shape (samples_num, z)
        :return: transformed z and the log_det
        """
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        if self.activation_func == "tanh":
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(
                self.w ** 2)

            def h_(x):
                return 1 / torch.cosh(x) ** 2
        elif self.activation_func == "leaky_relu":
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(
                self.w ** 2)  # constraint w.T * u neq -1, use >

            def h_(x):
                return (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        z_ = z + self.u * self.h(lin.unsqueeze(1))
        # print(self.u * self.h(lin.unsqueeze(1)))
        # print(torch.min(torch.sum(self.w * self.u) * h_(lin)))
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * self.u) * h_(lin)))
        return z_, torch.mean(log_det)


class ELBO(nn.Module):
    def __init__(self, log_p):
        super(ELBO, self).__init__()
        self.log_p = log_p

    def forward(self, z_k, log_det):
        return torch.mean(self.log_p(z_k)) - torch.sum(log_det)


class NF(nn.Module):
    def __init__(self, log_p, flows_name='planar', flows_num=2):
        super(NF, self).__init__()
        self.log_p = log_p
        self.flows_name = flows_name
        self.flows_num = flows_num
        if flows_name == 'planar':
            self.flows = nn.ModuleList([PlanarNet(input_dim=(2,)).to('cuda') for i in range(flows_num)])
        elif flows_name == 'NICE':
            self.flows = nn.ModuleList([NICENet(input_dim=(2,)).to('cuda') for i in range(flows_num)])

    def forward(self, z):
        loss = 0
        for k in range(self.flows_num):
            z, log_det = self.flows[k](z)
            loss += log_det
        return z, loss


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x):
        return -x


class NNReg(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, alpha=0.0):
        super(NNReg, self).__init__()
        super().__init__()
        self.input_dim = input_dim  # 输入维度
        self.output_dim = output_dim  # 输出维度
        self.dropout = dropout
        self.alpha = alpha  # LeakyRelu参数

        self.Dense1 = nn.Linear(input_dim, input_dim * 2)
        if input_dim >= 2:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim // 2)
            self.Dense3 = nn.Linear(input_dim // 2, output_dim)
        else:
            self.Dense2 = nn.Linear(input_dim * 2, input_dim)
        self.Dense3 = nn.Linear(input_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        x = self.leakyrelu(self.Dense1(x))
        x = self.leakyrelu(self.Dense2(x))
        x = self.leakyrelu(self.Dense3(x))
        return x


class NICENet(nn.Module):  # NICE
    """
    NICE: Additive Coupling Layers
    """

    def __init__(self, input_dim, d=1, device='cpu'):
        """
        :param input_dim: dimension of z, a tuple is needed
        :param d: partition num
        :param device:
        """
        super(NICENet, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.d = d
        assert d < input_dim[0]

        # initialize parameters
        self.nn = NNReg(input_dim=d, output_dim=input_dim[0] - d)
        self.s = nn.Parameter(torch.ones(input_dim))

    def forward(self, z):
        """
        :param z: current samples, with shape (samples_num, z)
        :return: transformed z and the log_det
        """
        z[:, self.d:] = z[:, self.d:] + self.nn(z[:, :self.d])
        z = z * self.s
        return z, torch.prod(self.s)


class RealNVPNet(nn.Module):  # NICE
    """
    NICE: Additive Coupling Layers
    """

    def __init__(self, input_dim, d=1, device='cpu'):
        """
        :param input_dim: dimension of z, a tuple is needed
        :param d: partition num
        :param device:
        """
        super(RealNVPNet, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.d = d
        assert d < input_dim[0]

        # initialize parameters
        self.nn_alpha = NNReg(input_dim=d, output_dim=input_dim[0] - d)
        self.nn_mu = NNReg(input_dim=d, output_dim=input_dim[0] - d)

    def forward(self, z):
        """
        :param z: current samples, with shape (samples_num, z)
        :return: transformed z and the log_det
        """
        alpha = self.nn_alpha(z[:, :self.d])
        z[:, self.d:] = z[:, self.d:] * alpha + self.nn_mu(z[:, :self.d])
        z = z * self.s
        return z, torch.exp(torch.sum(alpha))
