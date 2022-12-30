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


    def GreedyAttack(self, funccall, y, budget):
        #funccall: imput
        #y: true label
        self.model.eval()
        s_true = self.model(self.input_handle(funccall, y))[0]
        n_feature = funccall.shape[0]
        ## stage 1: find features
        print('start feature')
        s_diff_all = [] #store best score difference for every feature
        time_f_start = time.time()
        for i in range(n_feature):
            funccall_new = copy.deepcopy(funccall)
            s_diff_single = [] #store score difference for feature i
            for j in range(self.n_diagonosis_codes):
                funccall_new[i] = j
                funccall_new_v = self.input_handle(funccall_new, y)
                s_per = self.model(funccall_new_v)[0]
                diff_y = s_true[y] - s_per[y]
                s_diff_single.append(diff_y.cpu().detach().numpy().tolist())
            s_diff_all.append(max(s_diff_single)) #store the maximum change of score for each feature
        ind_per = np.argpartition(s_diff_all, -1*budget)[-1*budget:] #index of features to perturb
        time_f_end = time.time()
        print('feature time:', time_f_end-time_f_start)
        print(ind_per)

        ## stage 2: find worst pertuebation for features
        print('start perturb')
        time_per_start = time.time()
        ad_sample = copy.deepcopy(funccall)
        print('start perturbing')
        for i in ind_per: #traverse all perturbed features
            time_oneper_start = time.time()
            sf_diff = []
            for j in range(self.n_diagonosis_codes): #traverse all levels
                ad_sample[i] = j
                funccall_per = self.input_handle(ad_sample, y)
                s_per = self.model(funccall_per)[0]
                diff_y = s_true[y] - s_per[y]
                sf_diff.append(diff_y.cpu().detach().numpy().tolist())
            ind_best_per = np.argmax(sf_diff)
            print('best per:', ind_best_per)
            time_oneper_end = time.time()
            print('one perturbation time:', time_oneper_end-time_oneper_start)
            ad_sample[i] = ind_best_per
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