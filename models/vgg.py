import math
import torch
import torch.nn as nn
from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel
import numpy as np
from typing import cast


class VGG(BaseModel):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5) -> None:
        super(VGG , self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv_module = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                bn_module = nn.BatchNorm2d(v)
                conv_module, bn_module = ModuleInjection.make_prunable(conv_module, bn_module)
                layers += [conv_module, bn_module, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




def _vgg(cfg: str, batch_norm: bool, **kwargs) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model




cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}




def get_vgg_model(model, method ,num_classes):
    ModuleInjection.pruning_method = method
    ModuleInjection.prunable_modules = []
    if model == "vgg11":
        net= _vgg("A", True, num_classes = num_classes)
    if model == "vgg13":
        net =  _vgg("B", True,  num_classes = num_classes)
    if model == "vgg19":
        net = _vgg("E", True,  num_classes = num_classes)

    net.prunable_modules = ModuleInjection.prunable_modules
    return net




