import os
import argparse
import copy
import torch
from model import CharCNN
from data_loader import AGNEWs
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import time
import CGAttack4 as CGA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Character level CNN text classifier testing', formatter_class=argparse.RawTextHelpFormatter)
# model
parser.add_argument('--model-path', default='models_CharCNN/CharCNN2_epoch_37.pth.tar', help='Path to pre-trained acouctics model created by DeepSpeech training')
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--l0', type=int, default=1014, help='maximum length of input sequence to CNNs [default: 1014]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
# data
parser.add_argument('--test-path', metavar='DIR',
                    help='path to testing data csv', default='data/ag_news_csv/test.csv')
parser.add_argument('--batch-size', type=int, default=1, help='batch size for training [default: 128]')
parser.add_argument('--alphabet-path', default='alphabet.json', help='Contains all characters for prediction')
# device
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu' )
# logging options
parser.add_argument('--save-folder', default='Results/', help='Location to save epoch models')
# attack option
parser.add_argument('--itermax', default=200, type = int, help='max iteration of gradient ascent')
parser.add_argument('--budget', default = 5, type = int, help = 'number of perturbation')
parser.add_argument('--epsilon', default = 0.026, type = float, help = 'threshold for penalty')
parser.add_argument('--eval_num', default=500, type=int, help='the number of samples during evaluation')
parser.add_argument('--alpha', default = 1000, type = float, help='penalty parameter')
args = parser.parse_args()


budget = args.budget
epsilon = args.epsilon
eval_num = args.eval_num
alpha = args.alpha
itermax = args.itermax
print('budget:',budget)

# load data
test_dataset = AGNEWs(label_data_path=args.test_path, alphabet_path=args.alphabet_path)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
num_samples = test_dataset.__len__()
print('Data loaded.')
print('Test size:', num_samples)
#load model
args.num_features = len(test_dataset.alphabet)
model = CharCNN(args)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('Model loaded.')

attacker = CGA.Attacker(model, itermax)
print('Attacker created')


succ_rates = []
prob_rates = []
time_start = time.time()
changed_nodes = []
time_atk = []

for i_batch, (data) in enumerate(test_loader):
    if i_batch <200:
        print(i_batch)
        inputs, target = data
        #inputs, target= next(iter(test_loader))
        inputs = inputs.to(device)
        target = (target-1).to(device)
        time1 = time.time()
        succ_rate, changed_node = attacker.CGattack(inputs, target, budget, epsilon, eval_num, alpha)
        time2 = time.time()
        succ_rates.append(succ_rate)
        changed_nodes = changed_nodes + changed_node
        time_atk.append(time2-time1)


print('success rate:', sum(succ_rates)/len(succ_rates))
print('avg changed nodes:', sum(changed_nodes)/len(changed_nodes))
print('avg time:', sum(time_atk)/len(time_atk))


# 0.01/1; 0.014/2; 0.018/3; 0.022/4; 0.026/5
