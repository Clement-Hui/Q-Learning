from typing import *

import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    def __init__(self, network_dims: Union[Tuple, List]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, dim in enumerate(network_dims):
            if i == len(network_dims) - 1:
                break
            layer = nn.Linear(dim, network_dims[i + 1])
            self.layers.append(layer)

    def forward(self, tensor):
        result = tensor
        for i, layer in enumerate(self.layers):
            result = layer(result)
            if i != len(self.layers) - 1:
                result = F.relu(result)
                #result = F.dropout(result)

        return result
