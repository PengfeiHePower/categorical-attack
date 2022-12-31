import os
from itertools import combinations
import copy
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Attacker(object):
    def __init__(self, model, n_candidates):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        self.n_candidates = n_candidates # number of feature levels

    def classify(self, inputs):  
        ### funccall must fit the size of model [n_feature,1,n_level]
        logit = self.model(inputs)
        pred = torch.argmax(logit, dim=1) #prediction from model
        return pred


    def GradientAttack(self, inputs, target, budget):
        self.model.eval()
        s_true = self.model(inputs)[0]
        n_feature = inputs.shape[-1]
        #n_feature = funccall.shape[0]
        ## stage 1: find features, using the gradients of each feature
        time_f_start = time.time()
        inputs_new = copy.deepcopy(inputs)
        inputs_new.requires_grad = True
        loss_f = F.nll_loss(self.model(inputs_new), target, size_average=False)
        grad_f = torch.autograd.grad(loss_f, inputs_new)[0].squeeze()
        #print('grad_f:', grad_f)
        grad_f_abs = torch.abs(grad_f)
        s_f = torch.max(grad_f_abs, dim=0).values
        #print('s_f:', s_f)
        s_f = s_f.cpu().detach().numpy()
        ind_per = np.argpartition(s_f, -1*budget)[-1*budget:] #index of features to perturb
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