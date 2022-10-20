import numpy as np
from collections import defaultdict
from .layers import PrunableBatchNorm2d
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from .latencyPredictor import LatencyPredictor

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.prunable_modules = []
        self.prev_module = defaultdict()
        # self.lpModel = LatencyPredictor()
#         self.next_module = defaultdict()
        pass
    
    def set_threshold(self, threshold):
        self.prune_threshold = threshold
    
    def load_latency_predictor_weights(self, weights_path):
        self.lpModel.load_state_dict(torch.load(weights_path , map_location=torch.device("cuda")))
    
    def latency_prediction(self, device):
        model_encoding = self.get_encoding_from_zeta().float().to("cuda")
        return self.lpModel(model_encoding)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def calculate_prune_threshold(self, Vc, budget_type = 'channel_ratio'):
        zetas = self.give_zetas()
        if budget_type in ['volume_ratio']:
            zeta_weights = self.give_zeta_weights()
            zeta_weights = zeta_weights[np.argsort(zetas)]
        zetas = sorted(zetas)
        if budget_type == 'volume_ratio':
            curr_budget = 0
            indx = 0
            while(curr_budget<(1.-Vc)):
                indx+=1
                curr_budget+=zeta_weights[indx]
            prune_threshold = zetas[indx]
        else:
            prune_threshold = zetas[int((1.-Vc)*len(zetas))]
        return prune_threshold
    
    def smoothRound(self, x, steepness=20.):
        return 1./(1.+torch.exp(-1*steepness*(x-0.5)))
    
    def n_remaining(self, m, steepness=20.):
        return (m.pruned_zeta if m.is_pruned else self.smoothRound(m.get_zeta_t(), steepness)).sum()
    
    def is_all_pruned(self, m):
        return self.n_remaining(m) == 0
    
    def get_remaining(self, steepness=20., budget_type = 'channel_ratio'):
        """return the fraction of active zeta_t (i.e > 0.5)""" 
        n_rem = 0
        n_total = 0
        for l_block in self.prunable_modules:
            if budget_type == 'volume_ratio':
                n_rem += (self.n_remaining(l_block, steepness)*l_block._conv_module.output_area)
                n_total += (l_block.num_gates*l_block._conv_module.output_area)
            elif budget_type == 'channel_ratio':
                n_rem += self.n_remaining(l_block, steepness)
                n_total += l_block.num_gates
            elif budget_type == 'parameter_ratio':
                k = l_block._conv_module.kernel_size[0]
                prev_total = 3 if self.prev_module[l_block] is None else self.prev_module[l_block].num_gates
                prev_remaining = 3 if self.prev_module[l_block] is None else self.n_remaining(self.prev_module[l_block], steepness) 
                n_rem += self.n_remaining(l_block, steepness)*prev_remaining*k*k
                n_total += l_block.num_gates*prev_total*k*k
            elif budget_type == 'flops_ratio':
                k = l_block._conv_module.kernel_size[0]
                output_area = l_block._conv_module.output_area
                prev_total = 3 if self.prev_module[l_block] is None else self.prev_module[l_block].num_gates
                prev_remaining = 3 if self.prev_module[l_block] is None else self.n_remaining(self.prev_module[l_block], steepness) 
                curr_remaining = self.n_remaining(l_block, steepness)
                n_rem += curr_remaining*prev_remaining*k*k*output_area + curr_remaining*output_area
                n_total += l_block.num_gates*prev_total*k*k*output_area + l_block.num_gates*output_area
        return n_rem/n_total

    def get_normalized_encoding(self, encoding_ , p):
        tensor_family = encoding_[:4]
        tensor_count = encoding_[4:5] / 1000
        tensor_zeta = encoding_[5:]
        tensor_zeta_zero = torch.nn.functional.relu(tensor_zeta)
        tensor_zeta_normalized = torch.nn.functional.normalize(tensor_zeta_zero, p=p, dim =0)
        tensor_final = torch.where(tensor_zeta < 0 , tensor_zeta , tensor_zeta_normalized)
        #tensor_final = tensor_zeta_normalized
        final = torch.cat((tensor_family , tensor_count , tensor_final) , 0)
        return final 

    # To-Do Zetas extract from n_remaning function and then zetas ko encoding form m chaiye
    # encoding = [model_type, n_remaning() har ek layer p]  encoding chaiye 0 1 form so as to learn from network check help paper
    def get_encoding_from_zeta(self , steepness = 20. , p = 2):
        encoding = torch.zeros(4 , requires_grad=True).to("cuda")
        # encoding = torch.zeros(4 , requires_grad=False).to("cuda")
        count = 0
        normal_cnn_list = torch.tensor([] , requires_grad = True).to("cuda")
        encoding.data[2] = 1
        # feature_extractor = self.features

        self.model_prune_dict = {}
        ch_ar = []
        for i, layer in enumerate(self.prunable_modules):
            # print(layer)
            out_ch = layer._conv_module.out_channels
            ch_ar.append(out_ch)
            normal_cnn_list = torch.cat((normal_cnn_list , self.n_remaining(layer , steepness).view(1).to("cuda")), 0) 
            count += 1

            self.model_prune_dict[f"layer_{i}"] = [layer, int(self.n_remaining(layer , steepness).view(1).item())]

        
        # for feature in feature_extractor:
        #     if isinstance(feature , nn.BatchNorm2d): # change hua to prunable batch
        #         normal_cnn_list = torch.cat((normal_cnn_list , self.n_remaining(feature , steepness).view(1).to("cuda")), 0) 
        #         count += 1
    
        normal_cnn_list = torch.cat((normal_cnn_list , torch.tensor(-3.0 , requires_grad = True ).view(1).to("cuda")) , 0)
        
        # classifier = self.classifier
        # for layer in classifier:
        #     if isinstance(layer , nn.Linear):
        #         normal_cnn_list = torch.cat((normal_cnn_list , torch.tensor(layer.out_features , requires_grad = True , dtype=torch.float).view(1).to("cuda")) , 0)
        #         count += 1
        encoding= torch.cat((encoding , torch.tensor(count , requires_grad=True , dtype=torch.float).view(1).to("cuda")) , 0)
        encoding = torch.cat((encoding , normal_cnn_list) , 0)
        
        left_over = 405 - len(encoding)
        encoding = torch.cat((encoding , torch.zeros(left_over , requires_grad=True).to("cuda")) , 0)
        
        return self.get_normalized_encoding(encoding , 2), self.model_prune_dict, ch_ar

    def give_zetas(self):
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.get_zeta_t().cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas

    def give_zeta_weights(self):
        zeta_weights = []
        for l_block in self.prunable_modules:
            zeta_weights.append([l_block._conv_module.output_area]*l_block.num_gates)
        zeta_weights = [z for k in zeta_weights for z in k ]
        return zeta_weights/np.sum(zeta_weights)

    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas = self.give_zetas()
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        plt.hist(zetas)
        plt.show()
        plt.savefig("zeta_t_distribution.png")
        return exactly_zeros, exactly_ones
    
    def get_crispnessLoss(self, device):
        """loss reponsible for making zeta_t 1 or 0"""
        loss = torch.FloatTensor([]).to(device)
        for l_block in self.prunable_modules:
            loss = torch.cat([loss, torch.pow(l_block.get_zeta_t()-l_block.get_zeta_i(), 2)])
        return torch.mean(loss).to(device)

    def prune(self, Vc, budget_type = 'channel_ratio', finetuning=False, threshold=None):
        """prunes the network to make zeta_t exactly 1 and 0"""

        if budget_type == 'parameter_ratio':
            zetas = sorted(self.give_zetas())
            high = len(zetas)-1
            low = 0
            while low<high:
                mid = (high + low)//2
                threshold = zetas[mid]
                for l_block in self.prunable_modules:
                    l_block.prune(threshold)
                self.remove_orphans()
                if self.params()<Vc:
                    high = mid-1
                else:
                    low = mid+1
        elif budget_type == 'flops_ratio':
            zetas = sorted(self.give_zetas())
            high = len(zetas)-1
            low = 0
            while low<high:
                mid = (high + low)//2
                threshold = zetas[mid]
                for l_block in self.prunable_modules:
                    l_block.prune(threshold)
                self.remove_orphans()
                if self.flops()<Vc:
                    high = mid-1
                else:
                    low = mid+1
        else:
            if threshold==None:
                self.prune_threshold = self.calculate_prune_threshold(Vc, budget_type)
                threshold = min(self.prune_threshold, 0.9)
                
        for l_block in self.prunable_modules:
            l_block.prune(threshold)

        if finetuning:
            self.remove_orphans()
            return threshold
        else:
            problem = self.check_abnormality()
            return threshold, problem

    def unprune(self):
        for l_block in self.prunable_modules:
            l_block.unprune()
    
    def prepare_for_finetuning(self, device, budget, budget_type = 'channel_ratio'):
        """freezes zeta"""
        self.device = device
        self(torch.rand(2,3,32,32).to(device))
        threshold = self.prune(budget, budget_type=budget_type, finetuning=True)
        if budget_type not in ['parameter_ratio', 'flops_ratio']:
            while self.get_remaining(steepness=20., budget_type=budget_type)<budget:
                threshold-=0.0001
                self.prune(budget, finetuning=True, budget_type=budget_type, threshold=threshold)
        return threshold      

    def get_params_count(self):
        total_params = 0
        active_params = 0
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                active_param, total_param = l_block.get_params_count()
                active_params+=active_param 
                total_params+=total_param
            if isinstance(l_block, nn.Linear):
                linear_params = l_block.weight.view(-1).shape[0]
                active_params+=linear_params
                total_params+=linear_params
        return active_params, total_params

    def get_volume(self):
        total_volume = 0.
        active_volume = 0.
        for l_block in self.prunable_modules:
                active_volume_, total_volume_ = l_block.get_volume()
                active_volume+=active_volume_ 
                total_volume+=total_volume_
        return active_volume, total_volume

    def get_flops(self):
        total_flops = 0.
        active_flops = 0.
        for l_block in self.prunable_modules:
                active_flops_, total_flops_ = l_block.get_flops()
                active_flops+=active_flops_ 
                total_flops+=total_flops_
        return active_flops, total_flops
    
    def get_channels(self):
        total_channels = 0.
        active_channels = 0.
        for l_block in self.prunable_modules:
                active_channels+=l_block.pruned_zeta.sum().item()
                total_channels+=l_block.num_gates
        return active_channels, total_channels
          
    def set_beta_gamma(self, beta, gamma):
        for l_block in self.prunable_modules:
            l_block.set_beta_gamma(beta, gamma)
    
    def latency_prune(self, Vc , lpModel , initial_latency):
        zetas = sorted(self.give_zetas())
        high = len(zetas)-1
        low = 0
        while low<high:
            mid = (high + low)//2
            threshold = zetas[mid]
            for l_block in self.prunable_modules:
                l_block.prune(threshold)
            with torch.no_grad():
                latency_ratio = lpModel(self.get_encoding_from_zeta()) / initial_latency
            if latency_ratio < Vc:
                high = mid-1
            else:
                low = mid+1
        for l_block in self.prunable_modules:
            l_block.prune(threshold)

        return threshold
    
    def check_abnormality(self):
        n_removable = self.removable_orphans()
        isbroken = self.check_if_broken()
        if n_removable!=0. and isbroken:
            return f'both rem_{n_removable} and broken'
        if n_removable!=0.:
            return f'removable_{n_removable}'
        if isbroken:
            return 'broken'
        
    def check_if_broken(self):
        for bn in self.prunable_modules:
            if bn.is_imp and bn.pruned_zeta.sum()==0:
                return True
        return False

    def pruned_model(self):

        for name, module in self.model_prune_dict.items():
        # for name, module in self.named_children():
            print(name, module)
            break

        for i, module in enumerate(self.named_children()):
            print(i, module)
            if hasattr(module, 'conv'):
                print(module.conv)
                self.modules[i].conv = module.conv
            break
        
            
            

 