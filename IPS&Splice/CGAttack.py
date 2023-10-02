import os
from itertools import combinations
import argparse
from models import *
from utils import *
import copy
import torch
import torch.nn as nn
import cvxpy
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = {
    'Splice': 3,
    'IPS': 3,
}


def Expect_GumbelSM_grad(model, prob, inputs, label, alpha, epsilon, iters = 500): ### modified

    model.train()
    celoss = nn.CrossEntropyLoss()

    prob2 = prob.repeat(1, iters, 1)
    z = nn.functional.gumbel_softmax(prob2, dim = 2, tau = 0.1, hard = False)

    label_tensor = torch.tensor([label] * iters).long().cuda()
    ce = celoss(model(z), label_tensor)

    ## penalized distance
    iter_inputs = inputs.repeat(1, iters, 1)
    dist = torch.sum(-torch.log(z+0.001) * iter_inputs, dim = 2)    ## cross entropy loss
    #print('dist.shape', dist.shape)
    #input(123)
    dist = torch.mean(F.relu(torch.mean(dist, axis = 0) - epsilon)) ## only penalize over bar = 1.5

    loss = ce - alpha * dist
    grad = torch.autograd.grad(loss, prob)[0]

    #print('CE Loss', ce.item(), 'Dist', dist.item(), 'Gradient Norm', torch.sum(torch.abs(grad)).item())
    return grad

#[feature, batch, level]



class Attacker(object):
    def __init__(self, Dataset, best_parameters_file, itermax=50, tau=0.1, lr=0.1):
        self.n_labels = num_classes[Dataset]
        if Dataset == 'Splice':
            self.model = geneRNN()
        elif Dataset == 'IPS':
            self.model = IPSRNN()
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.n_diagonosis_codes = num_category[Dataset] # number of feature levels
        self.lr = lr #learning rate
        self.itermax = itermax # gradient descending iterations
        self.tau = tau

    def input_handle(self, funccall, y):
        funccall = [funccall]
        y = [y]
        t_diagnosis_codes, _ = pad_matrix(funccall, y, self.n_diagonosis_codes)
        return torch.tensor(t_diagnosis_codes, device = device)

    def classify(self, funccall):
        logit = self.model(funccall)
        pred = torch.argmax(logit, dim=1) #prediction from model
        return pred


    def CGattack(self, funccall, y, budget, epsilon, eval_num, alpha):
        # epsilon: a set of thresholds for searching
        # alpha: a set of penalty parameters for searching
        #self.model.eval()
        # n_feature = funccall.shape[0]

        sample = self.input_handle(funccall, y)
        self.prob = torch.clone(sample) * 100
        self.prob.requires_grad = True

        for epsilon_s in epsilon:
            for k in range(self.itermax):
                print('pgd step ' +str(k))
                grad = Expect_GumbelSM_grad(self.model, self.prob, sample, y, alpha, epsilon_s)

                self.prob = self.prob + self.lr * torch.sign(grad)
                self.prob = torch.clip(self.prob, min = 1e-3, max = 15)
                self.prob.detach()
                self.prob.requires_grad_
            ## Evaluation
            self.model.eval()
            prob3 = self.prob.repeat(1, eval_num, 1)
            z = nn.functional.gumbel_softmax(prob3, dim = 2, tau = 0.1, hard = False)

            changed_nodes=[]
            outputs = []
            succ_rate = 0

            for j in range(eval_num):
                dist = torch.sum(torch.abs(z[:, j:j+1, :] - sample)) / 2
                output = torch.argmax(self.model(z[:, j:j+1, :]))

                changed_nodes.append(dist.item())
                outputs.append(output.item())

                if not (output == y):
                    print('True Label', y, 'After Attack', output.item(), 'Perturb #', dist.item())
                    if dist.item()<=budget:
                        succ_rate = 1
                        break

        print('avg changed nodes:', sum(changed_nodes)/len(changed_nodes))
        print('success rate:', succ_rate)

        return succ_rate, changed_nodes