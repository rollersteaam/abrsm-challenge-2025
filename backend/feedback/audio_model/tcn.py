import math
import torch 
import torch.nn.functional as F 
from torch import nn
from torch.nn.modules.normalization import LayerNorm
#from timm.models.layers import trunc_normal_, DropPath, to_2tuple

class conv_frontend(nn.Module):
    #Convolutional Frontend
    def __init__(self, in_channels, out_channels, dropout):
        #In_channels: Number of input channels
        #Out_channels: Number of output channels
        super().__init__()
        #Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = (3, 7), stride = (1,1), padding = (1, 3), groups = 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size = (12, 1), stride = (2,1), padding = (0, 0), groups = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3, 1))
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size = (3, 5), stride = (1,1), padding = (1, 2), groups = 1)
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3, 1))
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.batchnorm3(x)

        x = x.squeeze(2)
        x = x.transpose(2, 1)

        return x


class class_model(nn.Module):
    #Full Model
    def __init__(self, dim, drop, in_channels):
        super().__init__()


        self.tcn_block_0 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 2,dilation = 1)
        self.tcn_block_1 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 4,dilation = 2)
        self.tcn_block_2 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 8,dilation = 4)
        self.tcn_block_3 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 16,dilation = 8)
        self.tcn_block_4 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 32,dilation = 16)
        self.tcn_block_5 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 64,dilation = 32)
        self.tcn_block_6 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 128,dilation = 64)
        self.tcn_block_7 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 256,dilation = 128)
        self.tcn_block_8 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 512,dilation = 256)
        self.tcn_block_9 = nn.Conv1d(dim, dim, kernel_size = 5, stride = 1, padding = 1024,dilation = 512)
        self.tcn_block_10 = nn.Conv1d(dim,dim, kernel_size = 5, stride = 1, padding = 2048,dilation = 1024)

        self.conv0 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv9 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv10 = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)

        #Spectrogram Definitions
        self.conv_frontend = conv_frontend(1, dim, 0.1)

        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):
    
        x = self.conv_frontend(x)
        x = x.transpose(1, 2)

        # ...existing code for TCN blocks...
        x0 = self.tcn_block_0(x)
        x0 = self.sigmoid(x0) * self.tanh(x0)
        x0 = self.conv0(x0)
        x = x0 + x
        
        x1 = self.tcn_block_1(x)
        x1 = self.sigmoid(x1) * self.tanh(x1)
        x1 = self.conv1(x1)
        x =  x1 + x

        x2 = self.tcn_block_2(x)
        x2 = self.sigmoid(x2) * self.tanh(x2)
        x2 = self.conv2(x2)
        x = x2 + x

        # x3 = self.tcn_block_3(x)
        # x3 = self.sigmoid(x3) * self.tanh(x3)
        # x3 = self.conv3(x3)
        # x = x3 + x

        # x4 = self.tcn_block_4(x)
        # x4 = self.sigmoid(x4) * self.tanh(x4)
        # x4 = self.conv4(x4)
        # x = x4 + x
        
        # x5 = self.tcn_block_5(x)
        # x5 = self.sigmoid(x5) * self.tanh(x5)
        # x5 = self.conv5(x5)
        # x = x5 + x

        # x6 = self.tcn_block_6(x)
        # x6 = self.sigmoid(x6) * self.tanh(x6)
        # x6 = self.conv6(x6)
        # x = x6 + x
        
        # x7 = self.tcn_block_7(x)
        # x7 = self.sigmoid(x7) * self.tanh(x7)
        # x7 = self.conv7(x7)
        # x = x7 + x

        # x8 = self.tcn_block_8(x)
        # x8 = self.sigmoid(x8) * self.tanh(x8)
        # x8 = self.conv8(x8)
        # x = x8 + x
        
        # x9 = self.tcn_block_9(x)
        # x9 = self.sigmoid(x9) * self.tanh(x9)
        # x9 = self.conv9(x9)
        # x = x9 + x
        
        # x10 = self.tcn_block_10(x)
        # x10 = self.sigmoid(x10) * self.tanh(x10)
        # x10 = self.conv10(x10)
        # x = x10 + x
        x = x.transpose(1, 2)

        return x