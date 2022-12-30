import os
from itertools import combinations
import argparse
from models import *
from utils import *
import copy
import torch
import torch.nn as nn
import numpy as np
import time
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = {
    'Splice': 3,
    'IPS': 3,
}


class Attacker(object):
    def __init__(self, Dataset, best_parameters_file, log_f, itermax=50):
        self.n_labels = num_classes[Dataset]
        if Dataset == 'Splice':
            self.model = geneRNN()
        elif Dataset == 'IPS':
            self.model = IPSRNN()
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.log_f = log_f
        self.criterion = nn.CrossEntropyLoss()
        self.n_diagonosis_codes = num_category[Dataset] # number of feature levels
        self.itermax = itermax # gradient descending iterations

    def input_handle(self, funccall, y):  # input:funccall, output:(seq_len,n_sample,m)[i][j][k]=k,
        # (seq_len,n_sample,m)对一个[i][j]选中的[k]为0.96，其余[k']=0
        ##### keep it, one_hot of input
        funccall = [funccall]
        y = [y]
        t_diagnosis_codes, _ = pad_matrix(funccall, y, self.n_diagonosis_codes)
        return torch.tensor(t_diagnosis_codes, device = device) 
        #return a tensor n by 1 by p, n by 1 is original size, p is size of categorical feature

    def classify(self, funccall):  
        ### funccall must fit the size of model [n_feature,1,n_level]
        logit = self.model(funccall)
        logit = logit.cpu()
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy() #prediction from model
        return pred


    def GradientAttack(self, funccall, y, budget):
        #funccall: imput
        #y: true label
        self.model.train()
        s_true = self.model(self.input_handle(funccall, y))[0]
        #n_feature = funccall.shape[0]
        ## stage 1: find features, using the gradients of each feature
        time_f_start = time.time()
        funccall_new = copy.deepcopy(funccall)
        funccall_new_v = self.input_handle(funccall_new, y)
        funccall_new_v.requires_grad = True
        loss_f = self.criterion(self.model(funccall_new_v), torch.tensor([y], device = device))
        grad_f = torch.autograd.grad(loss_f, funccall_new_v)[0].view(funccall_new_v.shape[0],funccall_new_v.shape[2])
        grad_f_abs = torch.abs(grad_f)
        s_f = torch.max(grad_f_abs, dim=1).values
        s_f = s_f.cpu().detach().numpy()
        ind_per = np.argpartition(s_f, -1*budget)[-1*budget:] #index of features to perturb
        time_f_end = time.time()
        print('feature time:', time_f_end-time_f_start)
        print(ind_per)

        ## stage 2: find worst pertuebation for features
        time_per_start = time.time()
        ad_sample = copy.deepcopy(funccall)
        ad_sample_temp = copy.deepcopy(funccall)
        sublist = [list(range(self.n_diagonosis_codes))] * budget
        s_diff_best = -1 * np.inf
        print('start perturbing')
        for element in itertools.product(*sublist):
            ad_sample_temp[ind_per] = element
            funccall_per = self.input_handle(ad_sample_temp, y)
            s_per = self.model(funccall_per)[0]
            diff_y = s_true[y] - s_per[y]
            if diff_y > s_diff_best:
                ad_sample[ind_per] = element
                s_diff_best = diff_y
        time_per_end = time.time()
        print('pretrubation time:', time_per_end-time_per_start)
        return ad_sample #return this sample


    def eval_attack(self, ad_sample, y):
        # for each data, sample a set of adversrial examples and compute test accuracy
        # return a list of success rates
        self.model.eval()
        pred = self.classify(self.input_handle(ad_sample, y))
        succ_rate = [np.int((pred != y))]
        return succ_rate