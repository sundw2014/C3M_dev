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

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', type=int, default=1024)
parser.add_argument('--num_train', type=int, default=131072) # 4096 * 32
parser.add_argument('--num_test', type=int, default=32768) # 1024 * 32
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr_step', type=int, default=5)
parser.add_argument('--log', type=str)
parser.add_argument('--pretrained', type=str)

args = parser.parse_args()

os.system('cp *.py '+args.log)
os.system('cp -r models/ '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r systems/ '+args.log)

config = importlib.import_module('config_'+args.task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX

system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func

model = importlib.import_module('model_sysid')
get_model = model.get_model

model_fhat, fhat_func = get_model(num_dim_x, use_cuda=args.use_cuda)

if args.pretrained is not None:
    ck = torch.load(args.pretrained)
    model_fhat.load_state_dict(ck['model_fhat'])

# constructing datasets
def sample_x():
    return (X_MAX-X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_full():
    x = sample_x()
    return x

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

def forward(x):
    # x: bs x n x 1
    bs = x.shape[0]
    with torch.no_grad():
        f = f_func(x).detach()
    fhat = fhat_func(x)

    loss = 0
    loss += ((f - fhat).squeeze(-1)**2).sum(axis=1).mean()

    return loss

    
param_list = list(model_fhat.parameters())
optimizer = torch.optim.Adam(param_list, lr=args.learning_rate)

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
        x = []
        for id in indices[b*bs:(b+1)*bs]:
            if args.use_cuda:
                x.append(torch.from_numpy(X[id]).float().cuda())
            else:
                x.append(torch.from_numpy(X[id]).float())

        x = torch.stack(x).detach()

        start = time.time()

        loss = forward(x)

        start = time.time()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backwad(): %.3f s'%(time.time() - start))

        total_loss += loss.item() * x.shape[0]
    return total_loss / len(X)

best_loss = np.inf

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    loss = trainval(X_tr)
    print("Training loss: %.9f"%loss)
    loss = trainval(X_te)
    print("Epoch %d: Testing loss: %.9f"%(epoch, loss))

    if loss <= best_loss:
        best_loss = loss
        filename = args.log+'/model_best.pth.tar'
        torch.save({'args':args, 'precs':loss, 'model_fhat': model_fhat.state_dict()}, filename)
