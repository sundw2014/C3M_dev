import torch
from torch.autograd import grad
import torch.nn.functional as F

import importlib
import numpy as np
import time
from tqdm import tqdm

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

from np2pth import get_system_wrapper, get_controller_wrapper
from utils import EulerIntegrate

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', type=int, default=1024)
parser.add_argument('--num_train', type=int, default=4096) # 4096 * 32
parser.add_argument('--num_test', type=int, default=1024) # 1024 * 32
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr_step', type=int, default=5)
parser.add_argument('--log', type=str)

args = parser.parse_args()

os.system('cp *.py '+args.log)
os.system('cp -r models/ '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r systems/ '+args.log)

config = importlib.import_module('config_'+args.task)
time_bound = config.time_bound
time_step = config.time_step
time_points = config.t

system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
f_np, B_np, _, num_dim_x, num_dim_control = get_system_wrapper(system)

model = importlib.import_module('model_'+args.task)
get_model = model.get_model
_, _, model_u_w1, model_u_w2, _, u_func = get_model(num_dim_x, num_dim_control, use_cuda=args.use_cuda)

class ClosedLoopFunc(nn.Module):
    def __init__(self, xref, uref, t):
        super(ClosedLoopFunc, self).__init__()
        self.xref = xref
        self.uref = uref
        self.t = t.reshape(-1)

    def forward(self, t, x):
        idx_t = np.where((t.item() - self.t) >= 0)[0][-1]
        xstar = self.xref[:,idx_t,:,:]
        ustar = self.uref[:,idx_t,:,:]
        u = u_func(x, x-xstar, ustar)
        dotx = f_func(x) + B_func(x).matmul(u)

# constructing datasets
def sample_full():
    x_0, xstar_0, ustar = config.system_reset(np.random.rand())
    ustar = [u.reshape(-1,1) for u in ustar]
    xstar_0 = xstar_0.reshape(-1,1)
    xstar, _ = EulerIntegrate(None, f_np, B_np, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)
    return (x_0, np.array(xstar), np.array(ustar))

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

def forward(x0, xref, uref, t):
    # x: bs x n x 1
    # xref: bs x T x n x 1
    # uref: bs x T x m x 1
    bs = x0.shape[0]
    closed_loop_func = ClosedLoopFunc(xref, uref, t)
    x = odeint(closed_loop_func, x0, torch.from_numpy(t).float().type(x0.type())) # T x bs x n x 1

    loss = 0
    loss += torch.sqrt(((x.transpose(0,1) - xref)**2).sum(dim=2)).mean()

    return loss

optimizer = torch.optim.Adam(list(model_u_w1.parameters()) + list(model_u_w2.parameters()), lr=args.learning_rate)

def trainval(X, bs=args.bs, train=True): # trainval a set of x
    # torch.autograd.set_detect_anomaly(True)
    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0

    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x0 = []; xref = []; uref = [];
        for id in indices[b*bs:(b+1)*bs]:
            if args.use_cuda:
                x0.append(torch.from_numpy(X[id][0]).float().cuda())
                xref.append(torch.from_numpy(X[id][1]).float().cuda())
                uref.append(torch.from_numpy(X[id][2]).float().cuda())
            else:
                x0.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())

        x0, xref, uref = (torch.stack(d).detach() for d in (x0, xref, uref))

        start = time.time()

        loss = forward(x0, xref, uref, time_points)

        start = time.time()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backwad(): %.3f s'%(time.time() - start))

        total_loss += loss.item() * x0.shape[0]
    return total_loss / len(X)

best_loss = np.inf

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    loss = trainval(X_tr, train=True)
    print("Training loss: ", loss)
    loss = trainval(X_te, train=False)
    print("Epoch %d: Testing loss: "%epoch, loss)

    if loss < best_loss:
        best_loss = loss
        filename = args.log+'/model_best.pth.tar'
        filename_controller = args.log+'/controller_best.pth.tar'
        torch.save({'args':args, 'precs':(loss), 'model_u_w1': model_u_w1.state_dict(), 'model_u_w2': model_u_w2.state_dict()}, filename)
        torch.save(u_func, filename_controller)
