import torch

num_dim_x = 4
num_dim_control = 2

import numpy as np
from utils import temp_seed
with temp_seed(1024):
    freqs = np.array(range(1,10+1))
    weights = np.random.randn(len(freqs)*num_dim_x, num_dim_x)
    weights = torch.from_numpy(weights / np.sqrt((weights**2).sum(axis=0, keepdims=True)))
    freqs = torch.from_numpy(freqs.reshape(1,1,-1))

def disturbance(x):
    global weights, freqs
    weights = weights.type(x.type())
    freqs = freqs.type(x.type())
    features = (torch.sin(x) * freqs).reshape(x.shape[0],-1) # B x (n*F)
    disturbances = features.matmul(weights).unsqueeze(-1) # B x n x 1
    return disturbances * 0

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    px, py, theta, v = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v * torch.cos(theta)
    f[:, 1, 0] = v * torch.sin(theta)
    f[:, 2, 0] = 0
    f[:, 3, 0] = 0
    return f + disturbance(x)

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 2, 0] = 1
    B[:, 3, 1] = 1
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')
