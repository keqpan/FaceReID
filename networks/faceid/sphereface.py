import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

class sphere20a(nn.Module):
    def __init__(self,classnum=10574, feature=False, dn_block=None, return_features=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = AngleLinear(512,self.classnum)
        
        self.dn_block = dn_block
        self.return_features = return_features


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x_relu1_3 = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x_relu1_3))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x_relu2_5 = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x_relu2_5))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x_relu3_9 = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x_relu3_9))
        x_relu4_3 = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x_relu4_3.view(x_relu4_3.size(0),-1)
        x_emb = self.fc5(x)
        if self.dn_block is not None:
            x_emb = self.dn_block(x_emb)
        if self.feature: return x_emb

        x = self.fc6(x_emb)
        if self.return_features:
            return x, x_relu1_3, x_relu2_5, x_relu3_9, x_relu4_3, x_emb
        return x
    
class sphere20a_sr(nn.Module):
    def __init__(self,classnum=10574, feature=False, return_features=False):
        super(sphere20a_sr, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)
        
        self.conv1_3_2x = nn.Conv2d(64//4,64,3,1,1)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)
        
        self.conv2_5_2x = nn.Conv2d(128//4,64,3,1,1)
        self.relu2_5_2x = nn.PReLU(64)
        self.conv2_5_4x = nn.Conv2d(128//4,128//4,3,1,1)
        self.relu2_5_4x = nn.PReLU(128//4)
        self.res_4x_conv = nn.Conv2d(128//4,3,3,1,1)
        
        self.conv_fmap_2x = nn.Conv2d(128,128,3,1,1)
        self.relu_fmap_2x = nn.PReLU(128)
        
        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = AngleLinear(512,self.classnum)
        
        self.pixel_shuffle2x = nn.PixelShuffle(2)
        self.hardtanh = nn.Hardtanh()
#         self.return_features = return_features

    def block_relu2_5(self, x):
        x = self.relu1_1(self.conv1_1(x))
#         print("relu1_1:", x.shape)
        
        x_relu1_3 = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
#         print("relu1_3:", x_relu1_3.shape)
        x_relu1_3_2x = self.pixel_shuffle2x(x_relu1_3)
#         print("relu1_3_2x:", x_relu1_3_2x.shape)

        x = self.relu2_1(self.conv2_1(x_relu1_3))
#         print("relu2_1:", x.shape)
        
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x_relu2_5 = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
#         print("relu2_5:", x_relu2_5.shape)
        return x_relu2_5, x_relu1_3

    def forward(self, x):
        x_relu2_5_0, x_relu1_3 = self.block_relu2_5(x)
        x_relu2_5_2x = self.relu2_5_2x(self.conv2_5_2x(self.pixel_shuffle2x(x_relu2_5_0)))
#         print("relu2_5_2x:", x_relu2_5_2x.shape)
        feat_map_2x = self.relu_fmap_2x(self.conv_fmap_2x(torch.cat((x_relu1_3, x_relu2_5_2x), dim=1)))
#         print("feat_map_2x:", feat_map_2x.shape)
        feat_map_4x = self.relu2_5_4x(self.conv2_5_4x(self.pixel_shuffle2x(feat_map_2x)))
#         print("feat_map_4x:", feat_map_4x.shape)
        res_4x = self.res_4x_conv(feat_map_4x)
        sr = x-res_4x
        
        x_relu2_5, _ = self.block_relu2_5(sr)
        
        x = self.relu3_1(self.conv3_1(x_relu2_5))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x_relu3_9 = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x_relu3_9))
        x_relu4_3 = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x_relu4_3.view(x_relu4_3.size(0),-1)
        x_emb = self.fc5(x)
        if self.feature: return x_emb

        x = self.fc6(x_emb)
#         if self.return_features:
#             return x, x_relu1_3, x_relu2_5, x_relu3_9, x_relu4_3, x_emb
        return x, sr
