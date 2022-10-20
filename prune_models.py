import argparse
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm_notebook

from utils import *
from models import get_model
from datasets import DataManager
from models.latencyPredictor import LatencyPredictor

import logging
from datetime import datetime
from time import time
import json

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed_everything(43)

ap = argparse.ArgumentParser(description='pruning with heaviside continuous approximations and logistic curves')
ap.add_argument('dataset', choices=['c10', 'c100', 'tin','svhn'], type=str, help='Dataset choice')
ap.add_argument('model', type=str, help='Model choice')
ap.add_argument('--budget_type', choices=['channel_ratio', 'volume_ratio','parameter_ratio','flops_ratio'], default='channel_ratio', type=str, help='Budget Type')
ap.add_argument('--Vc', default=0.25, type=float, help='Budget Constraint')
ap.add_argument('--batch_size', default=8, type=int, help='Batch Size')
ap.add_argument('--epochs', default=20, type=int, help='Epochs')
ap.add_argument('--workers', default=8, type=int, help='Number of CPU workers')
ap.add_argument('--valid_size', '-v', type=float, default=0.1, help='valid_size')
ap.add_argument('--lr', default=0.001, type=float, help='Learning rate')
ap.add_argument('--test_only','-t', default=False, type=bool, help='Testing')

ap.add_argument('--decay', default=0.001, type=float, help='Weight decay')

# need to do hyperparameter search on these 4 parameters
ap.add_argument('--w1', default=30., type=float, help='weightage to budget loss')
ap.add_argument('--w2', default=10., type=float, help='weightage to crispness loss')
ap.add_argument('--b_inc', default=5., type=float, help='beta increment')
ap.add_argument('--g_inc', default=2., type=float, help='gamma increment')

ap.add_argument('--cuda_id', '-id', type=str, default='0', help='gpu number')
args = ap.parse_args()

valid_size = args.valid_size
BATCH_SIZE = args.batch_size
Vc = torch.FloatTensor([args.Vc])

# logging.basicConfig(filename=f"./logs/pruning_{args.w1}_{args.w2}_{args.b_inc}_{args.g_inc}.log",
#         format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#                         datefmt='%m/%d/%Y %H:%M:%S',
#                         level=logging.INFO)

# logger.info(args)

# ############################### preparing dataset ################################

data_object = DataManager(args)
trainloader, valloader, testloader = data_object.prepare_data()
dataloaders = {
        'train': trainloader, 'val': valloader, "test": testloader
}

############################### preparing model ###################################


# model2 = model.pruned_model()

models_dict = {}
from tqdm import tqdm

for model_def in ['r50', 'r101','r110', 'r152', 'r32', 'r18', 'r56', 'r20']:

    print(model_def)
    model = get_model(model_def, 'prune', data_object.num_classes, data_object.insize)
    model.to(device)

    model.prune(0.5)
    initial_model_encoding, model_prune_dict, ch_ar = model.get_encoding_from_zeta()

    models_arr = []
    for j in tqdm(range(5000)): # change range for different number of models
        layer2ch = {}
        for i, layer in enumerate(model_prune_dict.keys()):
            layer2ch[layer] = int(np.random.randint(ch_ar[i]*0.5, ch_ar[i] + ch_ar[i]*0.5, 1)[0])

        models_arr.append(layer2ch)

    models_dict[model_def] = models_arr

    # Serializing json
    json_object = json.dumps(models_dict, indent=4)

    # Writing to sample.json
    with open("models_defs.json", "w") as outfile:
        outfile.write(json_object)

