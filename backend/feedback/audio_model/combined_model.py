import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm

class combined_model(nn.Module):
    #Full Model
    def __init__(self,drop):
        super().__init__()

        self.emb_size = 6144

        self.mark_linear1 = nn.Linear(self.emb_size+1, 512)
        self.mark_linear2 = nn.Linear(512, 256)
        self.mark_linear3 = nn.Linear(256, 40)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, y):

        #X Shape = (B,C,H,W)
        mark_vector = nn.Flatten()(y)
        mark_vector = self.mark_linear1(mark_vector)
        mark_vector = self.dropout(mark_vector)
        mark_vector = F.gelu(mark_vector)
        mark_vector = self.mark_linear2(mark_vector)
        mark_vector = self.dropout(mark_vector)
        mark_vector = F.gelu(mark_vector)
        mark_vector = self.mark_linear3(mark_vector)
        return mark_vector