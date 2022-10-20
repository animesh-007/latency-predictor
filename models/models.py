from .resnet import get_resnet_model
from .prune_resnet import get_prune_resnet_model
from .network_slimming_resnet import get_network_slimming_model
from .mobilenet import get_mobilenet
from .vgg import get_vgg_model

def get_model(model, method, num_classes, insize, model_prune_dict=None):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, model_name
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable model
    """

    if model in ['wrn', 'r50', 'r101','r110', 'r152', 'r32', 'r18', 'r56', 'r20']:
        if model_prune_dict:
            net = get_prune_resnet_model(model, method, num_classes, insize, model_prune_dict)
        else:
            net = get_resnet_model(model, method, num_classes, insize)
    elif model in ['r164']:
        net = get_network_slimming_model(method, num_classes)
    elif model in ['mobilenetv2']:
        net = get_mobilenet(model, method, num_classes)
    elif model in ['vgg11', 'vgg13' ,'vgg19']:
        net = get_vgg_model(model , method, num_classes)
    return net