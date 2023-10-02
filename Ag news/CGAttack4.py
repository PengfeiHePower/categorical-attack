import os
from itertools import combinations
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
import itertools
import datetime
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Expect_GumbelSM_grad(model, prob, inputs, label, alpha, epsilon, iters = 200): ### modified

    celoss = nn.CrossEntropyLoss()
    true_idx = torch.sum(inputs, dim = 1, keepdim = True)

    prob2 = prob.repeat(iters, 1, 1)
    z = nn.functional.gumbel_softmax(torch.log(prob2+1e-5), dim = 1, tau = 0.1, hard = False)

    ## caculate the gradient
    label_tensor = torch.tensor([label] * iters).long().cuda()
    z = z.type(torch.cuda.FloatTensor) * true_idx.repeat(iters,1,1)
    ce = celoss(model(z), label_tensor)
    ##penalty
    iter_inputs = inputs.repeat(iters,1,1)
    dist = torch.sum(-torch.log(z+1e-5) * iter_inputs, dim = 1) #CE loss
    #dist = torch.sum(z * iter_inputs, dim = 1)
    dist = torch.mean(F.relu(torch.mean(dist, axis = 1) - epsilon))

    loss = ce - alpha * dist
    grad = torch.autograd.grad(loss, prob)[0]
    print('CE Loss', ce.item(), 'Dist', dist.item(), 'Gradient Norm', torch.sum(torch.abs(grad)).item())


    return grad



class Attacker(object):
    def __init__(self, model, itermax=50, lr=0.001):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        self.lr = lr #learning rate
        self.itermax = itermax # gradient descending iterations



    def CGattack(self, inputs, y, budget, epsilon, eval_num, alpha):
        #self.model.train()
        #n_feature = inputs.shape[1]
        true_idx = torch.sum(inputs, dim = 1, keepdim = True)
        self.prob = torch.clone(inputs) * 10
        self.prob.requires_grad = True

        for k in range(self.itermax):

            print('pgd step ' +str(k))
            grad = Expect_GumbelSM_grad(self.model, self.prob, inputs, y, alpha, epsilon)

            self.prob = self.prob + self.lr * torch.sign(grad) * true_idx
            self.prob = torch.clip(self.prob, min = 1e-3, max = 15)
            #print('self.prob:', self.prob)
            self.prob.detach()
            self.prob.requires_grad_()
            self.prob.retain_grad()
            #input(123)

            ## Evaluation
        self.model.eval()
        prob3 = self.prob.repeat(eval_num, 1, 1)
        z = nn.functional.gumbel_softmax(torch.log(prob3 + 1e-5), dim = 1, tau = 0.1, hard = True)
        true_idx_z = true_idx.repeat(eval_num, 1, 1)
        z = z * true_idx_z

        changed_nodes=[]
        outputs = []

        for j in range(eval_num):
            dist = torch.sum(torch.abs(z[j:j+1, :, :] - inputs)) / 2
            output = torch.argmax(self.model(z[j:j+1, :, :]))

            changed_nodes.append(dist.item())
            outputs.append(output.item())

            if not (output == y):
                print('True Label', y, 'After Attack', output.item(), 'Perturb #', dist.item())

        succ_atk = [i for i in range(eval_num) if changed_nodes[i]<=budget and outputs[i]!=y]
        succ_rate = int(len(succ_atk)>0)

        print('avg changed nodes:', sum(changed_nodes)/len(changed_nodes))
        print('success rate:', succ_rate)

        print('Sampling Finished')
        return succ_rate, changed_nodes