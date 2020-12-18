import torch
from torch import nn

def get_model(num_dim_x, use_cuda = False):
    model_f = torch.nn.Sequential(
        torch.nn.Linear(num_dim_x, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_x, bias=False))

    if use_cuda:
        model_f = model_f.cuda()

    def f_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)
        return model_f(x).unsqueeze(-1)

    return model_f, f_func
