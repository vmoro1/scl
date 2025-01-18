import torch
from torch.nn import functional as F
from collections import OrderedDict


class SinusoidalActivation(torch.nn.Module):
    def __init__(self):
        super(SinusoidalActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class MLP(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False):
        super(MLP, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = SinusoidalActivation
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    