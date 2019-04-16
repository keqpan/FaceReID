import sys
sys.path.append("/home/safin/")

import torch.nn as nn
import torch as th

from pydl.networks.UDNet.net import UDNet
from pydl.networks.ResDNet.net import ResDNet

from pydl.nnLayers import modules
from pydl.nnLayers import init
from pydl.utils import loadmat
from pydl.nnLayers.cascades import nconv2D, nconv_transpose2D
from pydl.nnLayers.functional.functional import L2Proj
from pydl.utils import formatInput2Tuple, getPad2RetainShape
from pydl.nnLayers.functional import functional as F

import torch
from torch.nn.utils import weight_norm


"""
ResDNet with swaped projection like in UDNet
"""
class UDNetPA(nn.Module):
    
    def __init__(self, kernel_size,\
                 input_channels,\
                 output_features,\
                 convWeightSharing = True,\
                 pad = 'same',\
                 padType = 'symmetric',\
                 conv_init = 'dct',\
                 bias_f= True,\
                 bias_t = True,\
                 scale_f = True,\
                 scale_t = True,\
                 normalizedWeights = True,\
                 zeroMeanWeights = True,\
                 alpha = True,\
                 rpa_depth = 5,\
                 rpa_kernel_size1 = (3,3),\
                 rpa_kernel_size2 = (3,3),\
                 rpa_output_features = 64,\
                 rpa_init = 'msra',\
                 rpa_bias1 = True,\
                 rpa_bias2 = True,\
                 rpa_prelu1_mc = True,\
                 rpa_prelu2_mc = True,\
                 prelu_init = 0.1,\
                 rpa_scale1 = True,\
                 rpa_scale2 = True,\
                 rpa_normalizedWeights = True,\
                 rpa_zeroMeanWeights = True,\
                 shortcut = (True,False),\
                 clb = 0,\
                 cub = 255):

        super(UDNetPA, self).__init__()
        
        kernel_size = formatInput2Tuple(kernel_size,int,2)
        
        if isinstance(pad,str) and pad == 'same':
            pad = getPad2RetainShape(kernel_size)
            pad = (pad[1], pad[2])
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))              
        
        self.pad = formatInput2Tuple(pad,int,2)
        self.padType = padType
        self.normalizedWeights = normalizedWeights
        self.zeroMeanWeights = zeroMeanWeights
        self.convWeightSharing = convWeightSharing

        # Initialize conv weights
        shape = (output_features,input_channels)+kernel_size
        self.conv_weights = nn.Parameter(th.Tensor(th.Size(shape)))
        init.convWeights(self.conv_weights,conv_init)

        # Initialize the scaling coefficients for the conv weight normalization
        if scale_f and normalizedWeights:
            self.scale_f = nn.Parameter(th.Tensor(output_features).fill_(1))
        else:
            self.register_parameter('scale_f', None)       
        
        # Initialize the bias for the conv layer
        if bias_f:
            self.bias_f = nn.Parameter(th.Tensor(output_features).fill_(0))
        else:
            self.register_parameter('bias_f', None)            

        # Initialize the bias for the transpose conv layer
        if bias_t:
            self.bias_t = nn.Parameter(th.Tensor(input_channels).fill_(0))
        else:
            self.register_parameter('bias_t', None)                      

        if not self.convWeightSharing:
            self.convt_weights = nn.Parameter(th.Tensor(th.Size(shape)))
            init.convWeights(self.convt_weights,conv_init)      

            if scale_t and normalizedWeights:
                self.scale_t = nn.Parameter(th.Tensor(output_features).fill_(1))
            else:
                self.register_parameter('scale_t', None)           
        
        
        numparams_prelu1 = output_features if rpa_prelu1_mc else 1
        numparams_prelu2 = rpa_output_features if rpa_prelu2_mc else 1
        
        self.rpa_depth = rpa_depth
        self.shortcut = formatInput2Tuple(shortcut,bool,rpa_depth,strict = False)
        self.resPA = nn.Sequential(*nn.ModuleList([modules.ResidualPreActivationLayer(\
                        rpa_kernel_size1,rpa_kernel_size2,output_features,\
                        rpa_output_features,rpa_bias1,rpa_bias2,1,1,\
                        numparams_prelu1,numparams_prelu2,prelu_init,padType,\
                        rpa_scale1,rpa_scale2,rpa_normalizedWeights,\
                        rpa_zeroMeanWeights,rpa_init,self.shortcut[i]) \
                        for i in range(self.rpa_depth)]))
        
        self.bbproj = nn.Hardtanh(min_val = clb, max_val = cub)  
        
        # Initialize the parameter for the L2Proj layer
        if alpha:
            self.alpha = nn.Parameter(th.Tensor(1).fill_(0))
        else:
            self.register_parameter('alpha',None)
        
    def forward(self, input, stdn, net_input = None):
        if net_input is None:
            net_input = input
        
        output = nconv2D(input,self.conv_weights,bias=self.bias_f,stride=1,\
                     pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
#         for m in self.resPA:
#             output = m(output)
        output = self.resPA(output)
        
        if self.convWeightSharing:
            output = nconv_transpose2D(output,self.conv_weights,bias=self.bias_t,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_f,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)
        else:
            output = nconv_transpose2D(output,self.convt_weights,bias=self.bias_t,\
                     stride=1,pad=self.pad,padType=self.padType,dilation=1,\
                     scale=self.scale_t,normalizedWeights=self.normalizedWeights,
                     zeroMeanWeights=self.zeroMeanWeights)            

        output = F.L2Prox.apply(input-output,net_input,self.alpha,stdn)
        return self.bbproj(output)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'depth = ' + str(self.rpa_depth) \
            + ', convWeightSharing = ' + str(self.convWeightSharing)\
            + ', shortcut = ' + str(self.shortcut) + ')' 
    

import torch.nn.functional as F
import torch.nn.init as init
    
class L2Proj(nn.Module):
    # L2Prox layer
    # Y = NN_L2TRPROX(X,EPSILON) computes the Proximal map layer for the
    #   indicator function :
    #
    #                      { 0 if ||X|| <= EPSILON
    #   i_C(D,EPSILON){X}= {
    #                      { +inf if ||X|| > EPSILON
    #
    #   X and Y are of size H x W x K x N, and EPSILON = exp(ALPHA)*V*STDN
    #   is a scalar or a 1 x N vector, where V = sqrt(H*W*K-1).
    #
    #   Y = K*X where K = EPSILON / max(||X||,EPSILON);
    # s.lefkimmiatis@skoltech.ru, 22/11/2016.
    # pytorch implementation filippos.kokkinos@skoltech.ru 1/11/2017

    def __init__(self):
        super(L2Proj, self).__init__()

    def forward(self, x, stdn, alpha):
        if x.is_cuda:
            x_size = torch.cuda.FloatTensor(1).fill_(x.shape[1] * x.shape[2] * x.shape[3])
        else:
            x_size = torch.Tensor([x.shape[1] * x.shape[2] * x.shape[3]])
        numX = torch.sqrt(x_size-1)
        if x.is_cuda:
            epsilon = torch.cuda.FloatTensor(x.shape[0],1,1,1).fill_(1) * (torch.exp(alpha) * stdn * numX).view(-1,1,1,1)
        else:
            epsilon = torch.zeros(x.size(0),1,1,1).fill_(1) * (torch.exp(alpha) *  stdn * numX).view(-1,1,1,1)
        x_resized = x.view(x.shape[0], -1)
        x_norm = torch.norm(x_resized, 2, dim=1).view(x.size(0),1,1,1)
        max_norm = torch.max(x_norm, epsilon)
        result = x * (epsilon / max_norm)
        return result
    


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, weightnorm=None, shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.relu1 = nn.PReLU(num_parameters=planes,init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)


    def forward(self, x):
        out = self.relu1(x)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv1(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out,(1,1,1,1),'reflect')
        out = self.conv2(out)
        out = out[:,:, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out

import numpy as np


class ResNet_Den(nn.Module):

    def __init__(self, layer_size, color=True, weightnorm=None):
        self.inplanes = 64
        block = BasicBlock
        super(ResNet_Den, self).__init__()
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)

        # inntermediate layer has D-2 depth
        self.layer1 = self._make_layer(block, 64, layer_size)
        self.conv_out = nn.ConvTranspose2d(64, in_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        self.l2proj = L2Proj()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights = np.sqrt(2/(9.*64))*np.random.standard_normal(m.weight.data.shape)
                #weights = np.random.normal(size=m.weight.data.shape,
                #                           scale=np.sqrt(1. / m.weight.data.shape[1]))
                m.weight.data = torch.Tensor(weights)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.zeromean()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    def forward(self, x, stdn, alpha):
        self.zeromean()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.conv_out(out)
        out = self.l2proj(out, stdn, alpha)
        return out

class ResNet_Den_another(nn.Module):

    def __init__(self, layer_size, color=True, weightnorm=None):
        self.inplanes = 64
        super(ResNet_Den, self).__init__()
        block = BasicBlock
        if color:
            in_channels = 3
        else:
            in_channels = 1

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=0,
                               bias=True)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)

        # inntermediate layer has D-2 depth
        self.layer1 = self._make_layer(block, 64, layer_size)
        self.conv_out = nn.ConvTranspose2d(64, in_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        if weightnorm:
            self.conv_out = weight_norm(self.conv_out)

        self.l2proj = L2Proj()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weights = np.sqrt(2/(9.*64))*np.random.standard_normal(m.weight.data.shape)
                #weights = np.random.normal(size=m.weight.data.shape,
                #                           scale=np.sqrt(1. / m.weight.data.shape[1]))
                m.weight.data = torch.Tensor(weights)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.zeromean()
        
        self.alpha = torch.FloatTensor([2.]).cuda()
        self.alpha.requires_grad_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, weightnorm=True, shortcut=False))

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, weightnorm=True, shortcut=True))
        return nn.Sequential(*layers)

    def zeromean(self):
        # Function zeromean subtracts the mean E(f) from filters f
        # in order to create zero mean filters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    def forward(self, x, stdn):
        self.zeromean()
        out = F.pad(x,(2,2,2,2),'reflect')
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.conv_out(out)
        out = self.l2proj(out, stdn, self.alpha)
        return out
