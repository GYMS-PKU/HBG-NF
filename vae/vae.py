# Copyright (c) 2021 Dai HBG


"""
Implementation of vae.

log
2021-12-16
-- init
"""


import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, h_dim: int = 512, z_dim: int = 2):
        """
        :param input_dim:
        :param h_dim: hidden dimension
        :param z_dim: dimension of hidden parameters
        """
        super(VAE, self).__init__()

        self.Dense1 = nn.Linear(input_dim, h_dim)
        self.Dense_mu = nn.Linear(h_dim, z_dim)  # mu
        self.Dense_sigma = nn.Linear(h_dim, z_dim)  # log_sigma

        self.Dense2 = nn.Linear(z_dim, h_dim)
        self.Dense3 = nn.Linear(h_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.Dense1(x))
        mu = self.Dense_mu(h)
        log_sigma = self.Dense_sigma(h)
        return mu, log_sigma

    @staticmethod
    def reparameterize( mu, log_sigma):
        std = torch.exp(log_sigma * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x):
        h = torch.relu(self.Dense2(x))
        res = torch.sigmoid(self.Dense3(h))
        return res

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sampled_z = self.reparameterize(mu, log_sigma)
        res = self.decode(sampled_z)
        return res, mu, log_sigma


class MyVAELoss(nn.Module):
    def __init__(self):
        super(MyVAELoss, self).__init__()

    def forward(self, y, x, mu, log_sigma):
        reconstruction_loss = F.binary_cross_entropy(y, x, reduction='sum')
        divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)
        loss = reconstruction_loss + divergence
        return loss


from queue import Queue
from threading import Thread


class CudaDataLoader:
    def __init__(self, loader, device, queue_size=10):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        torch.cuda.set_device(self.device)
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) == str:
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def next(self):
        return self.__next__()

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset