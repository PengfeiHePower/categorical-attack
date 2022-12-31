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
        grad_f = torch.autograd.grad(loss_f, inputs_new)[0].squeeze() # compute gradient based on original input
        #print('grad_f:', grad_f)
        grad_f_abs = torch.abs(grad_f) #compute absolute value of gradients
        s_f = torch.max(grad_f_abs, dim=0).values #for each feature, find largest gradients of candidates
        print('s_f:', s_f)
        s_f = s_f.cpu().detach().numpy()
        ind_per = np.argpartition(s_f, -1*budget)[-1*budget:] #find indexes of features with largest gradients
        time_f_end = time.time()
        print('feature time:', time_f_end-time_f_start)
        print(ind_per)

        ## stage 2: find worst pertuebation for features
        time_per_start = time.time()
        ad_sample = copy.deepcopy(inputs)
        print('start perturbing')
        for i in ind_per: #search each feature for best perturbation
            #time_oneper_start = time.time()
            sf_diff = []
            for j in range(self.n_candidates): #traverse all candidates of feature
                onehot_element = F.one_hot(torch.tensor(j), self.n_candidates).float().to(device) #replace with each candidates
                ad_sample[0,:,i] = onehot_element
                s_per = self.model(ad_sample)[0]
                diff_y = s_true[target] - s_per[target] #compute the change of predicting score of true label
                sf_diff.append(diff_y.cpu().detach().numpy().tolist())
            #print('sf_diff:', sf_diff)
            ind_best_per = np.argmax(sf_diff) #choose candidate with largest drop of predicting score
            ad_sample[0,:,i] = F.one_hot(torch.tensor(ind_best_per), self.n_candidates).float().to(device)
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