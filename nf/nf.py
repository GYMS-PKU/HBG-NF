# Copyright (c) 2021 Dai HBG


"""
Implementation of basic nf methods.

log
2021-12-16
-- init
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


import sys
sys.path.append('C:/Users/Administrator/Desktop/华为云盘/学习资料/大四上/统计模型与计算方法/作业/第四次/code/HBG-NF/flows')


from flows import *


class NF(nn.Module):  # nf
    def __init__(self, flows_name: str = 'planar', flows_num: int = 2, device: str = 'cpu'):
        """
        :param flows_name: flows method
        :param flows_num: layers of flows
        :param device:
        """
        super(NF, self).__init__()
        self.flows_name = flows_name
        self.flows_num = flows_num
        self.device = device
        if flows_name == 'planar':
            self.flows = nn.ModuleList([PlanarNet(input_dim=(2,)).to(device) for _ in range(flows_num)])
        elif flows_name == 'NICE':
            self.flows = nn.ModuleList([NICENet(input_dim=(2,), order='first', device=device).to(device)
                                        if (i >= flows_num // 3) and (i <= flows_num * 2 // 3) else
                                        NICENet(input_dim=(2,), order='last', device=device).to(device)
                                        for i in range(flows_num)])
            # self.flows = nn.ModuleList([NICENet(input_dim=(2,), order='last').to('cuda')
            # for i in range(flows_num)])
        elif flows_name == 'RealNVP':
            self.flows = nn.ModuleList([RealNVPNet(input_dim=(2,), order='first', device=device).to(device)
                                        if (i >= flows_num // 3) and (i <= flows_num * 2 // 3) else
                                        RealNVPNet(input_dim=(2,), order='last', device=device).to(device)
                                        for i in range(flows_num)])
            # self.flows = nn.ModuleList([RealNVPNet(input_dim=(2,), order='last').to('cuda')
            # for i in range(flows_num)])

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
