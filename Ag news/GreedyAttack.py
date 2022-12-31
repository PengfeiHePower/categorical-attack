import os
from itertools import combinations
import copy
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
import time


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
        print(inputs.shape)
        s_true = self.model(inputs)[0]
        print('s_true:', s_true)
        n_feature = inputs.shape[-1]
        true_feature_ind = ((inputs.sum(dim=1).squeeze() !=0).nonzero(as_tuple=True)[0])

        ## stage 1: find features
        print('start feature')
        s_diff_all = [] #store best score difference for every feature
        time_f_start = time.time()
        for i in true_feature_ind: #traverse all features
            inputs_new = inputs.clone().detach()
            s_diff_single = [] #store score difference for feature i
            for j in range(self.n_candidates): #replace original feature value with every candidate value
                #print('j:', j)
                #print('self.n_candidates', self.n_candidates)
                onehot_element = F.one_hot(torch.tensor(j), self.n_candidates).float().to(device)
                inputs_new[0,:,i] = onehot_element #[batch_size, n_candidate, n_feature]
                s_per = self.model(inputs_new)[0]
                # print('s_per:', s_per)
                # print('s_per shape:', s_per.shape)
                # print('s_true:', s_true)
                # print('s_true shape:', s_true.shape)
                diff_y = s_true[target] - s_per[target] #compute the drop of predicting score for the true label
                #print('diff_y:', diff_y)
                s_diff_single = s_diff_single + diff_y.cpu().detach().tolist()
            s_diff_all.append(max(s_diff_single)) #store the maximum change of score for each feature
        #print(s_diff_all)
        ind_per = np.argpartition(s_diff_all, -1*budget)[-1*budget:] #choose features with largest drop of score
        #print(ind_per)
        #input(123)
        time_f_end = time.time()
        print('feature time:', time_f_end-time_f_start)
        print(ind_per)

        ## stage 2: find worst pertuebation for features
        time_per_start = time.time()
        ad_sample = copy.deepcopy(inputs)
        print('start perturbing')
        for i in ind_per: #search each feature for best perturbation
            time_oneper_start = time.time()
            sf_diff = []
            for j in range(self.n_candidates): #traverse all candidates of feature
                onehot_element = F.one_hot(torch.tensor(j), self.n_candidates).float().to(device)
                ad_sample[0,:,i] = onehot_element
                s_per = self.model(ad_sample)[0]
                diff_y = s_true[target] - s_per[target] #compute the change of predicting score of true label
                sf_diff.append(diff_y.cpu().detach().numpy().tolist())
            ind_best_per = np.argmax(sf_diff) #choose candidate with largest drop of predicting score
            time_oneper_end = time.time()
            print('one perturbation time:', time_oneper_end-time_oneper_start)
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