'''
DGE module is implemented as Gradient_Net and the inference is speed up by pytorch F.conv2d

loss_gd calculates the merge loss and gd_loss as well

Author: Ninghe Liu (lnh20@mails.tsinghua.edu.cn)

'''

################################################################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class loss_mse(nn.Module):
    def __init__(self):
        super(loss_mse, self).__init__()
    def forward(self, pred, truth):

        c = pred.shape[1]
        h = pred.shape[2]
        w = pred.shape[3]
        pred = pred.view(-1, c * h * w)
        truth = truth.view(-1, c * h * w)

        return torch.mean(torch.mean((pred - truth)**2, 1), 0)


# implement the DGE module in our CNN network
class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()

        kernel_x = [[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


# compute the merge loss
class loss_gd(nn.Module):
    def __init__(self):
        super(loss_gd, self).__init__()
    def forward(self, pred, truth):

        gradient_model = Gradient_Net()
        pdg = gradient_model(pred)
        tdg = gradient_model(truth)

        cp = pred.shape[1]
        hp = pred.shape[2]
        wp = pred.shape[3]
        pred = pred.view(-1, cp * hp * wp)
        truth = truth.view(-1, cp * hp * wp)

        c = pdg.shape[1]
        h = pdg.shape[2]
        w = pdg.shape[3]
        pdg = pdg.view(-1, c * h * w)
        tdg = tdg.view(-1, c * h * w)

        lossm = torch.mean(torch.mean((pred - truth)**2, 1), 0)
        lossg = torch.mean(torch.mean((pdg - tdg)**2, 1), 0)
        merge_loss = lossm + 0.2*lossg

        return merge_loss, lossg

# load index for training and validation dataset
def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/data/trainIdxs.txt'
    test_lists_path = current_directoty + '/data/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(test_lists) * 0.5)

    test_lists = test_lists[val_start_idx:-1]
    val_lists = test_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists


# test the tensor size
if __name__ == '__main__':
    loss = loss_gd()
    x = torch.zeros(1, 1, 160, 128).cuda()

    y = torch.ones(1, 1, 160, 128).cuda()

    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    r = loss(x, y)
    print(r)
