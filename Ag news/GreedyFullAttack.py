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


class Attacker(object):
    def __init__(self, model, n_candidates):
        self.n_candidates = n_candidates # number of feature levels
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(device)


    def classify(self, inputs):  
        ### funccall must fit the size of model [n_feature,1,n_level]
        logit = self.model(inputs)
        pred = torch.argmax(logit, dim=1) #prediction from model
        return pred


    def GreedyAttack(self, inputs, target, budget):
        self.model.eval()
        s_true = self.model(inputs)[0]
        n_feature = inputs.shape[-1]
        ## stage 1: find features
        s_diff_all = [] #store best score difference for every feature
        time_f_start = time.time()
        for i in range(n_feature):
            inputs_new = copy.deepcopy(inputs)
            s_diff_single = [] #store score difference for feature i
            for j in range(self.n_candidates):
                inputs_new[0,:,i] = F.one_hot(torch.tensor(j),self.n_candidates).float()
                s_per = self.model(inputs_new)[0]
                # print('s_per:', s_per)
                # print('s_per shape:', s_per.shape)
                # print('s_true:', s_true)
                # print('s_true shape:', s_true.shape)
                diff_y = s_true[target] - s_per[target]
                #print('diff_y:', diff_y)
                s_diff_single = s_diff_single + diff_y.cpu().detach().tolist()
            s_diff_all.append(max(s_diff_single)) #store the maximum change of score for each feature
        ind_per = np.argpartition(s_diff_all, -1*budget)[-1*budget:] #index of features to perturb
        time_f_end = time.time()
        print('feature time:', time_f_end-time_f_start)
        print(ind_per)

        ## stage 2: find worst pertuebation for features
        time_per_start = time.time()
        ad_sample = copy.deepcopy(inputs)
        ad_sample_temp = copy.deepcopy(inputs)
        sublist = [list(range(self.n_candidates))] * budget
        s_diff_best = -1 * np.inf
        print('start perturbing')
        for element in itertools.product(*sublist):
            onehot_element = torch.transpose(F.one_hot(torch.tensor(element),self.n_candidates),0,1).float().to(device)
            ad_sample_temp[0,:,ind_per] = onehot_element
            s_per = self.model(ad_sample_temp)[0]
            diff_y = s_true[target] - s_per[target]
            if diff_y > s_diff_best:
                ad_sample[0,:,ind_per] = onehot_element
                s_diff_best = diff_y
        time_per_end = time.time()
        print('pretrubation time:', time_per_end-time_per_start)
        return ad_sample #return this sample


    def eval_attack(self, ad_sample, y):
        # for each data, sample a set of adversrial examples and compute test accuracy
        # return a list of success rates
        self.model.eval()
        pred = self.classify(ad_sample)
        succ_rate = [np.int((pred != y))]
        return succ_rate