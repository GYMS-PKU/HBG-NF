# Copyright (c) 2021 Dai HBG


"""
Implementation of basic nf methods.

log
2021-12-09
-- init
2021-12-16
-- modify planar
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class PlanarNet(nn.Module):  # planar flow
    """
    f(z) = z + u * h(w * z + b)
    """

    def __init__(self, input_dim, u=None, w=None, b=None,
                 activation_func='tanh', device='cpu'):
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
            nn.init.uniform_(self.u, -0.1, 0.1)
        else:
            self.u = u
        if w is None:
            self.w = nn.Parameter(torch.empty(input_dim)[None])
            nn.init.uniform_(self.w, -0.1, 0.1)
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

    def _make_invertible(self):
        u = self.u
        w = self.w
        w_dot_u = torch.mm(u, w.t())
        if w_dot_u.item() >= -1.0:
            return

        norm_w = w / torch.norm(w, p=2, dim=1) ** 2
        bias = -1.0 + F.softplus(w_dot_u)
        u = u + (bias - w_dot_u) * norm_w
        self.u.data = u

    def forward(self, z):
        """
        :param z: current samples, with shape (samples_num, z)
        :return: transformed z and the log_det
        """
        self._make_invertible()
        if self.activation_func == "tanh":
            w_dot_u = torch.mm(self.u, self.w.T)
            affine = torch.mm(z, self.w.T) + self.b

            z = z + self.u * torch.tanh(affine)
            log_det = torch.log(1.0 + w_dot_u / torch.cosh(affine) ** 2)
            return z, torch.mean(log_det)

        elif self.activation_func == "leaky_relu":
            pass


class NICENet(nn.Module):  # NICE
    def __init__(self, input_dim: tuple = (2,), d: int = 1, order: str = 'first', device: str = 'cpu'):
        """
        :param input_dim: dimension of z
        :param d: dimensions that freeze
        :param order: first or last d dimension to be frozen
        :param device:
        """
        super(NICENet, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.d = d
        assert d < input_dim[0]

        # initialize parameters
        self.order = order
        if order == 'first':
            self.nn = NNRegNICE(input_dim=d, output_dim=input_dim[0] - d)
        else:
            self.nn = NNRegNICE(input_dim=input_dim[0] - d, output_dim=d)
        self.s = nn.Parameter(torch.ones(input_dim))

    def forward(self, z):
        """
        :param z: current samples, with shape (samples_num, z)
        :return: transformed z and the log_det
        """
        z_ = torch.zeros(z.shape).to(self.device)
        if self.order == 'first':
            z_[:, :self.d] = z[:, :self.d]
            z_[:, self.d:] = z[:, self.d:] + self.nn(z[:, :self.d])
        else:
            z_[:, self.d:] = z[:, self.d:]
            z_[:, :self.d] = z[:, :self.d] + self.nn(z[:, self.d:])
        z_ = z_ * self.s
        return z_, torch.sum(torch.log(self.s))


class RealNVPNet(nn.Module):
    def __init__(self, input_dim: tuple = (2,), d: int = 1, order: str = 'first', device: str = 'cpu'):
        """
        :param input_dim: dimension of z
        :param d: dimensions that freeze
        :param order: first or last d dimension to be frozen
        :param device:
        """
        super(RealNVPNet, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.d = d
        assert d < input_dim[0]

        # initialize parameters
        self.order = order
        if order == 'first':
            self.nn_alpha = NNReg(input_dim=d, output_dim=input_dim[0] - d)
            self.nn_mu = NNReg(input_dim=d, output_dim=input_dim[0] - d)
        else:
            self.nn_alpha = NNReg(input_dim=input_dim[0] - d, output_dim=d)
            self.nn_mu = NNReg(input_dim=input_dim[0] - d, output_dim=d)

    def forward(self, z):
        """
        :param z: current samples, with shape (samples_num, z)
        :return: transformed z and the log_det
        """
        z_ = torch.zeros(z.shape).to(self.device)
        if self.order == 'last':
            alpha = self.nn_alpha(z[:, self.d:])
            z_[:, self.d:] = z[:, self.d:]
            z_[:, :self.d] = z[:, :self.d] * torch.exp(alpha) + self.nn_mu(z[:, self.d:])
        else:
            alpha = self.nn_alpha(z[:, :self.d])
            z_[:, :self.d] = z[:, :self.d]
            z_[:, self.d:] = z[:, self.d:] * torch.exp(alpha) + self.nn_mu(z[:, :self.d])
        return z_, torch.mean(torch.sum(alpha, dim=1))


class NNReg(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, alpha=0.2):
        super(NNReg, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha

        self.Dense1 = nn.Linear(input_dim, input_dim * 2)
        self.Dense2 = nn.Linear(input_dim * 2, input_dim * 2)
        self.Dense3 = nn.Linear(input_dim * 2, output_dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = torch.tanh

    def forward(self, x):
        x = self.tanh(self.Dense1(x))
        x = self.leakyrelu(self.Dense2(x))
        x = self.Dense3(x)
        return x


class NNRegNICE(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, alpha=0.2):
        super(NNRegNICE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha

        self.Dense1 = nn.Linear(input_dim * 2, input_dim * 4)
        self.Dense2 = nn.Linear(input_dim * 4, input_dim * 2)
        self.Dense3 = nn.Linear(input_dim * 2, output_dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.tanh = torch.tanh

    def forward(self, x):
        x = torch.cat((x, x ** 2), dim=1)
        x = self.leakyrelu(self.Dense1(x))
        x = self.tanh(self.Dense2(x))
        x = self.Dense3(x)
        return x
