import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from tcn import class_model as tcn

class combined_model(nn.Module):
    #Full Model
    def __init__(self, dim, drop, in_channels_mask, in_channels_class, spec_size):
        super().__init__()

        self.sim2beats = tcn(dim, drop, in_channels_class)
        self.spec_size = spec_size
        
    def forward(self, y):
        #X Shape = (B,C,H,W)
        y = self.sim2beats(y)

        return mark_vector