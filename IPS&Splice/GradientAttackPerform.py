import os
from itertools import combinations
import argparse
import copy
import torch
import torch.nn as nn
import numpy as np
import time
import GradientAttack as GA

parser = argparse.ArgumentParser(description='IPS')
parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--dataset', default='IPS', type=str, help='dataset')
parser.add_argument('--modeltype', default='Normal', type=str, help='model type')
parser.add_argument('--time', default=60, type=int, help='time limit')
args = parser.parse_args()

def num_changed(true_samples, ad_samples):
    diff = true_samples != ad_samples
    num_diff = np.sum(diff)
    return num_diff

Dataset = args.dataset
Model_Type = args.modeltype
budget = args.budget
time_limit = args.time


print(Dataset, Model_Type, budget)
output_file = './Logs/%s/%s/' % (Dataset, Model_Type)
if os.path.isdir(output_file):
    pass
else:
    #os.mkdir(output_file)
    os.makedirs(output_file)

X, y = GA.load_data(Dataset)
print('Data loaded.')
best_parameters_file = GA.model_file(Dataset, Model_Type)
print('Model loaded.')
succ_rates = []

log_attack = open(
    './Logs/%s/%s/greedy_Attack%s.bak' % (Dataset, Model_Type, budget), 'w+')
attacker = GA.Attacker(Dataset, best_parameters_file, log_attack)
print('Attacker created')

ad_samples = []
time_start = time.time()
for i in range(10):
    print(i)
    print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

    sample = X[i]
    label = int(y[i])
    ad_sample = attacker.GradientAttack(sample, label, budget)
    ad_samples.append(ad_sample.tolist())
    succ_rate = attacker.eval_attack(ad_sample, label)
    succ_rates += succ_rate
    srs = sum(succ_rates)/len(succ_rates)
    print('success rate:', srs)


time_end = time.time()
srs = sum(succ_rates)/len(succ_rates)
changed_nodes = num_changed(X, np.array(ad_samples))
print('Total success rate:', srs)
print('Average changed node:', np.mean(changed_nodes))
print('total running time:', time_end - time_start)
print('Avg running time:', (time_end - time_start)/10)