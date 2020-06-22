import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os,sys,random,time
import argparse

from .focal_loss import *


start_time = time.time()
maxe = 0
for i in range(1000):
    x = torch.rand(12800, 2) * random.randint(1, 10)
    x = x.cuda()
    l = torch.rand(12800).ge(0.1).long()
    l = l.cuda()

    output0 = FocalLoss(gamma=0)(x, l)
    output1 = nn.CrossEntropyLoss()(x, l)
    a = output0.item()
    b = output1.item()
    maxe = max(abs(a - b), maxe)
print('time:',time.time() - start_time,'max_error:',maxe)


start_time = time.time()
maxe = 0
for i in range(100):
    x = torch.rand(128, 1000, 8, 4) * random.randint(1, 10)
    x = x.cuda()
    l = torch.rand(128, 8, 4) * 1000    # 1000 is classes_num
    l = l.long().cuda()

    output0 = FocalLoss(gamma=0)(x, l)
    output1 = nn.NLLLoss()(F.log_softmax(x, dim=1), l)
    a = output0.item()
    b = output1.item()
    maxe = max(abs(a - b), maxe)
print('time:', time.time() - start_time,'max_error:', maxe)