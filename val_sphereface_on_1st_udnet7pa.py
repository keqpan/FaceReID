#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.optim as optim
import numpy as np

from networks.faceid.sphereface import sphere20a

from transforms.noising import GaussianNoise
import torchvision
from torchvision import transforms

from utils import printoneline, dt

stop_flag = False
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    global stop_flag
    stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

from loss import AngleLoss
import itertools

import sys
sys.path.append("/home/safin/")
from pydl import utils as pydlutil

def test_w_denoiser(model, denoiser, dataloader, l2dist):
    model.eval()
    denoiser.eval()
    labels_arr = []
    distances_arr = []
    for batch_idx, data in tqdm.tqdm(enumerate(dataloader)):
        data_x, data_y, data_label = data
        data_x, data_y, data_label = data_x.cuda(), data_y.cuda(), data_label.cuda()
        
        data_x = 255*data_x
        sigma = pydlutil.wmad_estimator(data_x).cuda()
        denoised_x = denoiser(data_x, sigma)
        denoised_x = (denoised_x-127.5)/128
        
        data_y = 255*data_y
        sigma = pydlutil.wmad_estimator(data_y).cuda()
        denoised_y = denoiser(data_y, sigma)
        denoised_y = (denoised_y-127.5)/128

        out_x = model(denoised_x)
        out_y = model(denoised_y)
        
        out_dists = l2dist(out_x, out_y)
        
        distances_arr += out_dists.data.cpu().numpy().tolist()
        labels_arr += data_label.data.cpu().numpy().tolist()
    
    return np.asarray(distances_arr), np.asarray(labels_arr)

def test(model, dataloader, l2dist):
    model.eval()
    labels_arr = []
    distances_arr = []
    for batch_idx, data in enumerate(dataloader):
        data_x, data_y, data_label = data
        data_x, data_y, data_label = data_x.cuda(), data_y.cuda(), data_label.cuda()
        
        data_x = (data_x*255 - 127.5)/128
        data_y = (data_y*255 - 127.5)/128

        imglist = [data_x.data.cpu().numpy(), data_x.data.cpu().numpy()[:,:,:,::-1], data_y.data.cpu().numpy(), data_y.data.cpu().numpy()[:,:,:,::-1]]

        img = np.vstack(imglist)
        img = torch.from_numpy(img).float().cuda()
        output = model(img)
        f = output.data
        
        cur_batch_size = data_x.size(0)
        out_x, out_y = f[:cur_batch_size], f[2*cur_batch_size:3*cur_batch_size]
        out_dists = l2dist(out_x, out_y)
        
        distances_arr += out_dists.data.cpu().numpy().tolist()
        labels_arr += data_label.data.cpu().numpy().tolist()
    
    return np.asarray(distances_arr), np.asarray(labels_arr)


def eval_acc(threshold, dists, labels):
    accuracy = ((dists > threshold) == labels).mean()
    return accuracy

def find_best(thresholds, dists, labels):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, dists, labels)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold, best_acc
    
from datasets.lfw import LFWDataset

high_noise_std_arr = (np.arange(30, 55, 4)/255).tolist()
low_noise_std_arr = (np.arange(5, 29, 4)/255).tolist()

train_data_dir = "/tmp/CASIA-WebFace-sphereface/"

import torch.nn as nn
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
Module = nn.Module
import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

unfold = F.unfold

def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
 
    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)
 
    # N x [inC * kH * kW] x [outH * outW]
    cols = unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
 
    out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
 
    if bias is not None:
        out = out + bias.expand_as(out)
    return out


class Conv2dLocal(Module):
 
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
 
    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)

class FeatureDenoiser(nn.Module):
    def __init__(self, n_features=512, n_channels=10):
        super(FeatureDenoiser, self).__init__()
        self.conv1 = Conv2dLocal(n_features, 1, 1, n_channels, 1, 1, 0)
        self.prelu_1 = nn.PReLU(n_channels)
        self.conv2 = Conv2dLocal(n_features, 1, n_channels, n_channels, 1, 1, 0)
        self.prelu_2 = nn.PReLU(n_channels)
        self.conv3 = Conv2dLocal(n_features, 1, n_channels, 1, 1, 1, 0)
        self.n_channels = n_channels
        
    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], 1)
        x = x + self.conv3(self.prelu_2(self.conv2(self.prelu_1(self.conv1(x)))))
        x = x.view(x.size()[0],-1)
        return x
    
from networks.denoise.pydl import ResDNet
from networks.denoise.pydl import UDNet
from networks.denoise.pydl import UDNetPA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation script')
    parser.add_argument('-d', '--device', type=str, required=True,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    
    denoiser = ResDNet(kernel_size = (5, 5),
                input_channels = 3,
                output_features = 32,
                rpa_depth = 7,
                shortcut=(False,True))
    denoiser_ckpt = torch.load("/home/safin/ckpt/1st_udnetpa/weigths_31")
    denoiser.load_state_dict(denoiser_ckpt)
    denoiser = denoiser.cuda()
    
    faceid = sphere20a(feature=True)
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/sphereface_01.01_high_noise/01.01_sphere20a_40.pth" #trained for high noise
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/faceid_joint_3stages_20.01_8"
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/faceid_joint_19.01_11" #trained for high noise

#     faceid_ckpt_path = "ckpt/1stage_udnet_fixed_sphereface_27.01/faceid/faceid_1stage_udnet_fixed_sphereface_27.01_30"
#     faceid_ckpt_path = "ckpt/1stage_udnet_fixed_sphereface_28.01_finetune/faceid/faceid_1stage_udnet_fixed_sphereface_28.01_finetune_5"
#     faceid_ckpt_path = "ckpt/joint_1stage_udnet_27.01_fixed/faceid/faceid_joint_1stage_udnet_27.01_fixed_60"
#     faceid_ckpt_path = "ckpt/joint_02.02/faceid/joint_02.02_32"
#     faceid_ckpt_path = "ckpt/joint_02.02_fixed/faceid/jo int_02.02_fixed_"+str(n_ckpt)
#     faceid_ckpt_path = "ckpt/joint_07.02_fixed/faceid/weights_"+str(n_ckpt)
    n_ckpt = 90
#     faceid_ckpt_path = "ckpt/sphereface_10.02/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/sphereface_14.02/faceid/weights_19" 0.984
    faceid_ckpt_path = "/home/safin/ckpt/sphereface_clean/sphere20a_19.pth"
    faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/1st_udnet7pa_sphereface_22.02/faceid/weights_70"
    faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/sphereface_on_1st_udnet7pa_27.02/faceid/weights_16"
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
    faceid.load_state_dict(torch.load(faceid_ckpt_path))
    faceid = faceid.cuda()

    basic_transform = transforms.Compose([
                             transforms.ToTensor()
                         ])
    noise_transform = transforms.Compose([
                             transforms.ToTensor(),
                             GaussianNoise(high_noise_std_arr, clamp=[0,1])
                         ])
    
    transform = noise_transform
    lfw_data_dir = "/home/safin/datasets/lfw/lfw-sphereface/"
    lfw_dataset = LFWDataset(lfw_data_dir, "/home/safin/datasets/lfw/pairsDevTrain.txt", transform, "png")
    dataloader_test = torch.utils.data.dataloader.DataLoader(lfw_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    
    print("The number of parameters:", sum(p.numel() for p in faceid.parameters()))

    l2dist = torch.nn.CosineSimilarity().cuda()
    dists, labels = test_w_denoiser(faceid, denoiser, dataloader_test, l2dist)
    thresholds = np.arange(-1.0, 1.0, 0.001)
    print("SphereFace, tested on high noised with denoiser:", find_best(thresholds, dists, labels))

    
    
#     exp = ExpRunner()
#     exp.init_model(args.device, last_ckpt=args.resume)
#     exp.run_experiments(args.name, args.epochs, batch_size=args.batch_size)
    
#     denoiser = UDNet(kernel_size = (5, 5),
#                   input_channels = 3,
#                   output_features = 74,
#                   rbf_mixtures = 51,
#                   rbf_precision = 4,
#                   stages = 1)
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/denoiser_joint_19.01_11"
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/model_udnet_3stages_17.01_4" #trained for high noise
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/denoiser_joint_3stages_20.01_8" 
#     denoiser_ckpt_path =  "/home/safin/FaceReID/ckpt/model_udnet_3stages_17.01_4"
#     denoiser_ckpt_path =  "/home/safin/FaceReID/ckpt/udnet_1stage_20.01/model_udnet_1stage_20.01_5"
#     denoiser_ckpt_path =  "ckpt/joint_02.02/denoiser/joint_02.02_35"
#     n_ckpt = 73
#     denoiser_ckpt_path = "ckpt/joint_02.02_fixed/denoiser/joint_02.02_fixed_"+str(n_ckpt)
#     denoiser_ckpt_path = "ckpt/joint_07.02_fixed/denoiser/weights_"+str(n_ckpt)
#     denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     freeze_model(denoiser)
#     denoiser = denoiser.cuda()
#     print("The number of denoiser parameters:", sum(p.numel() for p in denoiser.parameters()))
    
#     denoiser_ckpt_path = "/home/safin/pydl/networks/UDNet/ckpt/model_udnet_14.01_3"
#     denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     denoiser = denoiser.cuda()
    