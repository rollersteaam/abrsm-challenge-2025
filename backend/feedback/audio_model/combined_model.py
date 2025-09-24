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

        self.embed_linear = nn.Linear(self.emb_size+1, 4000)

        self.mark_linear1 = nn.Linear(4000, 512)
        self.mark_linear2 = nn.Linear(512, 256)
        self.mark_linear3 = nn.Linear(256, 40)

        self.word_vec_size = 300
        self.word_linear1 = nn.Linear(4000, 512)
        self.word_linear2 = nn.Linear(512, self.word_vec_size*5)

        self.dropout = nn.Dropout(drop)

    def forward(self, y):

        #X Shape = (B,C,H,W)
        y = self.embed_linear(y)
        y = self.dropout(y)
        y = F.gelu(y)
        mark_vector = self.mark_linear1(y)
        mark_vector = self.dropout(mark_vector)
        mark_vector = F.gelu(mark_vector)
        mark_vector = self.mark_linear2(mark_vector)
        mark_vector = self.dropout(mark_vector)
        mark_vector = F.gelu(mark_vector)
        mark_vector = self.mark_linear3(mark_vector)

        word_vector = self.word_linear1(y)
        word_vector = F.gelu(word_vector)  
        word_vector = self.dropout(word_vector)
        word_vector = self.word_linear2(word_vector)
        word_vector = self.dropout(word_vector)
        word_vector = word_vector.view(-1, 5, self.word_vec_size)

        return mark_vector, word_vector