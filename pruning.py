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

logging.basicConfig(filename=f"./logs/pruning_{args.w1}_{args.w2}_{args.b_inc}_{args.g_inc}.log",
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

logger.info(args)

############################### preparing dataset ################################

data_object = DataManager(args)
trainloader, valloader, testloader = data_object.prepare_data()
dataloaders = {
        'train': trainloader, 'val': valloader, "test": testloader
}

############################### preparing model ###################################

model = get_model(args.model, 'prune', data_object.num_classes, data_object.insize)
state = torch.load(f"checkpoints/{args.model}_{args.dataset}_pretrained.pth"  , map_location=torch.device("cuda")) 
model.load_state_dict(state['state_dict'], strict=False)

model.to(device)


lpModel = LatencyPredictor()
lpModel.load_state_dict(torch.load("best_model_rtx.pt" , map_location=torch.device("cuda")))
lpModel.eval()
lpModel.to(device)

model.prune(0.5)
initial_model_encoding, model_prune_dict, ch_ar = model.get_encoding_from_zeta()
# model2 = model.pruned_model()

models_dict = {}

models_arr = []
for j in range(10):
    layer2ch = {}
    for i, layer in enumerate(model_prune_dict.keys()):
        layer2ch[layer] = int(np.random.randint(ch_ar[i]*0.5, ch_ar[i] + ch_ar[i]*0.5, 1)[0])

    models_arr.append(layer2ch)

models_dict["Resnet50"] = models_arr

# Serializing json
json_object = json.dumps(models_dict, indent=4)

# Writing to sample.json
with open("resnet50.json", "w") as outfile:
    outfile.write(json_object)


model2 = get_model(args.model, 'prune', data_object.num_classes, data_object.insize, model_prune_dict=model_prune_dict)

out = model2(torch.randn(1,3,224,224))
initial_model_encoding = initial_model_encoding.float().to("cuda")

with torch.no_grad():
    intial_latency = lpModel(initial_model_encoding)





############################### preparing for pruning ###################################

if os.path.exists('logs') == False:
    os.mkdir("logs")

if os.path.exists('checkpoints') == False:
    os.mkdir("checkpoints")


weightage1 = args.w1 #weightage given to budget loss
weightage2 = args.w2 #weightage given to crispness loss
steepness = 10. # steepness of gate_approximator

def latency_prediction(model_encoding):
    model_encoding = model_encoding.float().to(device)
    return lpModel(model_encoding)



CE = nn.CrossEntropyLoss()

def criterion(model, y_pred, y_true):
    global steepness

    # ce_loss = CE(y_pred, y_true).to(device)
    ce_loss = CE(y_pred, y_true)
    crispness_loss =  model.get_crispnessLoss(device)

    # need to check budget loss
    model_encoding = model.get_encoding_from_zeta()
    latency = latency_prediction(model_encoding)
    
    latency_fraction = latency / intial_latency
    budget_loss = ((latency_fraction.to(device)-Vc.to(device))**2).to(device)
    
    return budget_loss*weightage1 + crispness_loss*weightage2 + ce_loss
    # return crispness_loss*weightage2 + ce_loss


# device = torch.device(f"cuda:{str(args.cuda_id)}")
#device= torch.device("cpu")
model.to(device)
Vc.to(device)


param_optimizer = list(model.named_parameters())
no_decay = ["zeta"]
optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.decay,'lr':args.lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.lr},
    ]
optimizer = optim.AdamW(optimizer_parameters)


def train(model, loss_fn, optimizer, epoch):
    global steepness
    model.train()

    counter = 0
    tk1 = tqdm_notebook(dataloaders['train'], total=len(dataloaders['train']))
    running_loss = 0
    
    for x_var, y_var in tk1:
        optimizer.zero_grad()
        counter +=1
        x_var = x_var.to(device=device)
        y_var = y_var.to(device=device)
        scores = model(x_var)

        
        loss = loss_fn(model, scores, y_var)
        
        loss.backward()
        
        running_loss+=loss.item()
        tk1.set_postfix(loss=running_loss/counter)
        # logger.info("Training (%d / %d Steps) (loss=%2.5f)" % (counter, len(dataloaders['train']), (running_loss/counter)))

        optimizer.step()
        steepness=min(60,steepness+5./len(tk1))

    return running_loss/counter

def test(model, loss_fn, optimizer, phase, epoch):
    model.eval()
    counter = 0
    tk1 = tqdm_notebook(dataloaders[phase], total=len(dataloaders[phase]))
    running_loss = 0
    running_acc = 0
    total = 0
    with torch.no_grad():
        for x_var, y_var in tk1:
            counter +=1
            x_var = x_var.to(device=device)
            y_var = y_var.to(device=device)
            scores = model(x_var)
            loss = loss_fn(model,scores, y_var)
            _, scores = torch.max(scores.data, 1)
            y_var = y_var.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            
            correct = (scores == y_var).sum().item()
            running_loss+=loss.item()
            running_acc+=correct
            total+=scores.shape[0]
            tk1.set_postfix(loss=(running_loss /counter), acc=(running_acc/total))
            # logger.info(f"loss {(running_loss /counter)}, acc {(running_acc/total)}")

    return running_acc/total

best_acc = 0
beta, gamma = 1., 2.
model.set_beta_gamma(beta, gamma)

remaining_before_pruning = []
remaining_after_pruning = []
valid_accuracy = []
pruning_accuracy = []
pruning_threshold = []
exact_zeros = []
exact_ones = []
name = f'{args.model}_{args.dataset}_{str(np.round(Vc.item(),decimals=6))}_{args.budget_type}_pruned'
if args.test_only == False:
    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1} / {args.epochs}')
        logger.info(f'Starting epoch {epoch + 1} / {args.epochs}')
        model.unprune()
        train(model, criterion, optimizer, epoch)
        print(f'[{epoch + 1} / {args.epochs}] Validation before pruning')
        acc = test(model, criterion, optimizer, "val", epoch)
        logger.info(f'[{epoch + 1} / {args.epochs}] Validation before pruning accuracy: {acc}')
        #remaining = model.get_remaining(steepness, args.budget_type).item()
        #remaining_before_pruning.append(remaining)
        valid_accuracy.append(acc)
        exactly_zeros, exactly_ones = model.plot_zt()
        exact_zeros.append(exactly_zeros)
        exact_ones.append(exactly_ones)
        print(f'[{epoch + 1} / {args.epochs}] Validation after pruning')
        threshold = model.latency_prune(args.Vc , lpModel , intial_latency)
        acc = test(model, criterion, optimizer, "val", epoch)
        logger.info(f'[{epoch + 1} / {args.epochs}] Validation after pruning acc {acc}')
        #remaining = model.get_remaining(steepness, args.budget_type).item()
        pruning_accuracy.append(acc)
        pruning_threshold.append(threshold)
        #remaining_after_pruning.append(remaining)
        
        # 
        beta=min(6., beta+(0.1/args.b_inc))
        gamma=min(256, gamma*(2**(1./args.g_inc)))
        model.set_beta_gamma(beta, gamma)
        print("Changed beta to", beta, "changed gamma to", gamma)
        logger.info(f"Changed beta to {beta} changed gamma to {gamma}")    
        
        if acc>best_acc:
            print("**Saving checkpoint**")
            best_acc=acc
            torch.save({
                "epoch" : epoch+1,
                "beta" : beta,
                "gamma" : gamma,
                "prune_threshold":threshold,
                "state_dict" : model.state_dict(),
                "accuracy" : acc,
            }, f"checkpoints/{name}.pth")

        df_data=np.array([ valid_accuracy, pruning_accuracy, pruning_threshold]).T
        # df = pd.DataFrame(df_data,columns = ['Valid accuracy', 'Pruning accuracy', 'Pruning threshold', 'problems'])
        df = pd.DataFrame(df_data,columns = ['Valid accuracy', 'Pruning accuracy', 'Pruning threshold'])
        df.to_csv(f"logs/{name}.csv")
    
    logger.info(f"Best accuracy {best_acc}")
    print(f"Best accuracy {best_acc}")
