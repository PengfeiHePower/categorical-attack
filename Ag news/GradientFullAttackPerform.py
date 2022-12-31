import os
from itertools import combinations
import argparse
import copy
import torch
from model import CharCNN
from data_loader import AGNEWs
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import time
import GradientFullAttack as GFA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Character level CNN text classifier testing', formatter_class=argparse.RawTextHelpFormatter)
# model
parser.add_argument('--model-path', default='models_CharCNN/CharCNN2_epoch_14.pth.tar', help='Path to pre-trained acouctics model created by DeepSpeech training')
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
parser.add_argument('--budget', default = 3, type = int, help = 'number of perturbation')
args = parser.parse_args()

def num_changed(true_samples, ad_samples):
    diff = (true_samples != ad_samples).sum(dim=1)
    num_diff = (diff != 0).sum(dim=1)
    return num_diff

budget = args.budget


print('budget:',budget)
# load data
test_dataset = AGNEWs(label_data_path=args.test_path, alphabet_path=args.alphabet_path)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
print('Data loaded.')
#load model
args.num_features = len(test_dataset.alphabet)
model = CharCNN(args)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('Model loaded.')

attacker = GFA.Attacker(model, args.num_features)
print('Attacker created')

succ_rates = []

ad_samples = []
time_start = time.time()
changed_nodes = []

for i_batch, (data) in enumerate(test_loader):
#for i_batch in range(1):
    while i_batch<100:
        print(i_batch)
        inputs, target = data
        #inputs, target= next(iter(test_loader))
        inputs = inputs.to(device)
        target = (target-1).to(device)
        ad_sample = attacker.GradientAttack(inputs, target, budget)
        num_nodes = num_changed(inputs, ad_sample)
        changed_nodes = changed_nodes + num_nodes.cpu().detach().tolist()
        ad_samples.append(ad_sample.tolist())
        succ_rate = attacker.eval_attack(ad_sample, target)
        succ_rates += succ_rate
        srs = sum(succ_rates)/len(succ_rates)
        print('success rate:', srs)

time_end = time.time()
srs = sum(succ_rates)/len(succ_rates)
print('Total success rate:', srs)
print('Average changed node:', np.mean(changed_nodes))
print('total running time:', time_end - time_start)
print('avg running time:', (time_end - time_start)/len(succ_rates))