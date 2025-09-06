import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm
from tcn import class_model as tcn

class combined_model(nn.Module):
    #Full Model
    def __init__(self, dim, drop, in_channels_class, spec_size):
        super().__init__()

        self.sim2beats = tcn(dim, drop, in_channels_class)
        self.spec_size = spec_size

        self.mark_linear1 = nn.Linear(self.spec_size*dim, 1024)
        self.mark_linear2 = nn.Linear(1024, 512)
        self.mark_linear3 = nn.Linear(512, 40)
        self.dropout = nn.Dropout(drop)
    def forward(self, y):
        #X Shape = (B,C,H,W)

        y = self.sim2beats(y)

        mark_vector = nn.Flatten()(y)
        mark_vector = self.mark_linear1(mark_vector)
        mark_vector = self.dropout(mark_vector)
        mark_vector = F.relu(mark_vector)
        mark_vector = self.mark_linear2(mark_vector)
        mark_vector = self.dropout(mark_vector)
        mark_vector = F.relu(mark_vector)
        mark_vector = self.mark_linear3(mark_vector)
        #mark_vector = torch.softmax(mark_vector, dim=1)
        return mark_vector