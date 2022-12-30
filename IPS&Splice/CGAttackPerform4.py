import os
import CGAttack4 as CGA
import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_changed(true_samples, ad_samples):
    # true_samples, ad_samples: list or np array
    diff = true_samples != ad_samples
    num_diff = np.sum(diff, axis=1)
    return num_diff


parser = argparse.ArgumentParser(description='splice')
parser.add_argument('--epsilon', default=2, type=float, help='threshold of penalty')
parser.add_argument('--budget', default=4, type=float, help='purturb budget')
parser.add_argument('--dataset', default='IPS', type=str, help='dataset')
parser.add_argument('--modeltype', default='Normal', type=str, help='model type')
parser.add_argument('--lr', default = 0.1, type=float, help='learning rate')
parser.add_argument('--itermax', default=300, type = int, help='max iteration of gradient ascent')
parser.add_argument('--eval_num', default=500, type=int, help='the number of samples during evaluation')
parser.add_argument('--alpha', default = 10, type = float, help='penalty parameter')
args = parser.parse_args()


Dataset = args.dataset
Model_Type = args.modeltype
budget = args.budget
eval_num = args.eval_num
alpha = args.alpha
epsilon = args.epsilon


print(Dataset, Model_Type, budget)

X, y = CGA.load_data(Dataset)
print('data size:', len(X))
print('Data loaded.')
best_parameters_file = CGA.model_file(Dataset, Model_Type)
print('Model loaded.')

succ_rates = []
srs_total = []


attacker = CGA.Attacker(Dataset, best_parameters_file, itermax = args.itermax, lr=args.lr)
print('Attacker created')


succ_rates = []
changed_nodes = []
time_atk = []
for i in range(1):
    print('data index:',i)
    inputs = X[i]
    print(X[i])
    label = y[i]
    time1 = time.time()
    succ_rate, changed_node = attacker.CGattack(inputs, label, budget, epsilon, eval_num, alpha)
    time2 = time.time()
    succ_rates.append(succ_rate)
    changed_nodes = changed_nodes + changed_node
    time_atk.append(time2-time1)
    #input(123)

print('success rate:', sum(succ_rates)/len(succ_rates))
print('avg changed nodes:', sum(changed_nodes)/len(changed_nodes))
print('avg time:', sum(time_atk)/len(time_atk))

#splice:0.325/1.0; 0.5/2.0; 0.7/3.0; 0.85/4.0; 0.95/5.0
#ips:0.35/1.0; 0.85/2.0; 1.5/3.0; 2/4; 5/5