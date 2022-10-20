import torch
import torch.nn as nn
import torchvision


class LatencyPredictor(nn.Module):
  def __init__(self):
    INPUT_NEURON =  405
    super().__init__()
    self.layers = nn.Sequential( nn.Linear(INPUT_NEURON , 160),
      nn.ReLU(),
      nn.Linear(160, 480),
      nn.ReLU(),
      nn.Linear(480,480) , nn.ReLU(),
      nn.Linear(480,480),nn.ReLU(),
      nn.Linear(480,256) , nn.ReLU(),
      nn.Linear(256,1))
    
  def forward(self , x):
    return self.layers(x)

    
