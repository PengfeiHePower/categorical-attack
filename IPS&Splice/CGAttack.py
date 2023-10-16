import os
from itertools import combinations
import argparse
from models import *
from utils import *
import copy
import torch
import torch.nn as nn
# import cvxpy
import numpy as np
# import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = {
    'Splice': 3,
    'IPS': 3,
}


def Expect_GumbelSM_grad(model, prob, inputs, label, alpha, epsilon, iters = 500):
    ### calculate expected gradient via gumble-softmax
    # iters: the number of gradients to be computed, then the average of these gradients is the expected gradients.
    # prob: current probability tensor
    # inputs: original inputs, used to compute panelty
    # label: original label, used to compute panelty
    # alpha: penalty coefficient
    # epsilon: perturbation budget

    model.train()
    celoss = nn.CrossEntropyLoss()

    prob2 = prob.repeat(1, iters, 1)
    z = nn.functional.gumbel_softmax(prob2, dim = 2, tau = 0.1, hard = False) #generate gumble-softmax samples

    label_tensor = torch.tensor([label] * iters).long().cuda()
    ce = celoss(model(z), label_tensor)

    ## penalized distance
    iter_inputs = inputs.repeat(1, iters, 1)
    dist = torch.sum(-torch.log(z+0.001) * iter_inputs, dim = 2)    ## cross entropy loss, +0.001 to avoid log(0)
    #print('dist.shape', dist.shape)
    #input(123)
    dist = torch.mean(F.relu(torch.mean(dist, axis = 0) - epsilon)) ## mean of losses, only penalize over bar = 1.5

    loss = ce - alpha * dist # final loss
    grad = torch.autograd.grad(loss, prob)[0] #calculate gradients

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
        # transform original inputs into one-hot vectors. 
        # funccall:original input, size (sequence_length, batch_size)
        # y: original label
        # return: one-hot vector, size (sequence_length, batch_size, feature_levels/vocubulary_size)
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

        sample = self.input_handle(funccall, y) #transform original samples to one-hot vectors.
        self.prob = torch.clone(sample) * 100 #initialize the probability matrix using the above one-hot vector
        self.prob.requires_grad = True 

        for epsilon_s in epsilon: #for each budget in the budget_list, for linear search
            for k in range(self.itermax): # projected gradient descent step
                print('pgd step ' +str(k))
                grad = Expect_GumbelSM_grad(self.model, self.prob, sample, y, alpha, epsilon_s) # first compute expected gradients

                self.prob = self.prob + self.lr * torch.sign(grad) # update the probability via PGD
                self.prob = torch.clip(self.prob, min = 1e-3, max = 15) # clip the probability to a range
                self.prob.detach()
                self.prob.requires_grad_
            ## Evaluation
            self.model.eval()
            prob3 = self.prob.repeat(1, eval_num, 1)
            z = nn.functional.gumbel_softmax(prob3, dim = 2, tau = 0.1, hard = False) #generate multiple samples from trained probability

            changed_nodes=[]
            outputs = []
            succ_rate = 0

            for j in range(eval_num): #for each sample, do the prediction and compute how many features are changed
                dist = torch.sum(torch.abs(z[:, j:j+1, :] - sample)) / 2 # number of changed features
                output = torch.argmax(self.model(z[:, j:j+1, :])) #prediction

                changed_nodes.append(dist.item())
                outputs.append(output.item())

                if not (output == y):
                    print('True Label', y, 'After Attack', output.item(), 'Perturb #', dist.item())
                    if dist.item()<=budget:
                        succ_rate = 1
                        break

        print('avg changed nodes:', sum(changed_nodes)/len(changed_nodes))
        print('success rate:', succ_rate)
        # if there exists one sample have wrong prediction and the number of changed features is within the budget, claim a success

        return succ_rate, changed_nodes