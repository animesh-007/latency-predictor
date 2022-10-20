"""
Script to compute latency and fps of a model
"""
import os
import argparse
import time

import torch
# from gluoncv.torch.model_zoo import get_model
# from gluoncv.torch.engine.config import get_cfg_defaults
from torchvision import models
from models import get_model
from datasets import DataManager
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FLOPs of a model.')
    parser.add_argument('dataset', choices=['c10', 'c100', 'tin','svhn'], type=str, help='Dataset choice')
    parser.add_argument('model', type=str, help='Model choice')
    parser.add_argument('--config-file', type=str, help='path to config file.')
    parser.add_argument('--batch_size', type=int, default=32, help='temporal clip length.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--num-runs', type=int, default=105,
                        help='number of runs to compute average forward timing. default is 105')
    parser.add_argument('--num-warmup-runs', type=int, default=5,
                        help='number of warmup runs to avoid initial slow speed. default is 5')

    args = parser.parse_args()
    

    # need to load the json file and read model name and defination
    # step 1 load json file
    # args.model = key from model_dict
    
  
    # Opening JSON file
    f = open('models_defs.json')
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    
    # step 2 load model layers info from json file
    # model_prune_dict = layers info from json file
    # {"r50":[[layers, latency], []}
    models_dict = {}
    for model_name in data.keys():
        model_arr = []
        for model_def in data[model_name]:
            print(model_def)
    
            model = get_model(model_name, 'prune', 100, 32, model_prune_dict=model_def)
    

            # model = models.resnet152(pretrained=True)
            model.eval()
            model.cuda()
            input_tensor = torch.rand(args.batch_size, 3, args.input_size, args.input_size, requires_grad=True).cuda()
            print('Model is loaded, start forwarding.')

            with torch.no_grad():
                for i in range(args.num_runs):
                    if i == args.num_warmup_runs:
                        start_time = time.time()
                    pred = model(input_tensor)

            end_time = time.time()
            total_forward = end_time - start_time
            print('Total forward time is %4.2f seconds' % total_forward)

            actual_num_runs = args.num_runs - args.num_warmup_runs
            latency = total_forward / actual_num_runs
            # fps = (cfg.CONFIG.DATA.CLIP_LEN * cfg.CONFIG.DATA.FRAME_RATE) * actual_num_runs / total_forward

            # print("FPS: ", fps, "; Latency: ", latency)
            print("Latency: ", latency)
            model_arr.append([model_def, latency])
            break

        models_dict[model_name] = model_arr
        break
    # Serializing json
    json_object = json.dumps(models_dict, indent=4)

    # Writing to sample.json
    with open("models_latencies.json", "w") as outfile:
        outfile.write(json_object)


    # need to store lateny and model defination in a json file