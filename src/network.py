"""Defines the class DeepVesselNetFCN."""

import torch
import torch.nn as nn
import torch.nn.init as init
import math

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
class GestureNetFCN(nn.Module):
    """INPUT - 3DCONV - 3DCONV - 3DCONV - 3DCONV - FCN """

    def __init__(self, nchannels=3, nlabels=35):
        """
        Builds the network structure with the provided parameters

        Input:
        - nchannels (int): number of input channels to the network
        - nlabels (int): number of labels to be predicted
        - dim (int): dimension of the network
        - batchnorm (boolean): sets if network should have batchnorm layers
        - dropout (boolean): set if network should have dropout layers
        """
        super().__init__()

        self.nchannels = nchannels
        self.nlabels = nlabels
                
        self.conv = nn.Sequential(nn.Conv3d(in_channels=self.nchannels, out_channels=5, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(5),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(in_channels=5, out_channels=10, kernel_size=5, padding=2),
                                  nn.BatchNorm3d(10),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(in_channels=10, out_channels=20, kernel_size=5, padding=2),
                                  nn.BatchNorm3d(20),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(in_channels=20, out_channels=50, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(50),
                                  nn.ReLU(inplace=True),
                               )
        self.fcn = nn.Linear(10500000, nlabels)
        
        self.sigmoid = nn.Sigmoid()

        

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)        
        x = self.fcn(x)
#         x = self.sigmoid(x)
        
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)