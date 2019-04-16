#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.optim as optim
import numpy as np

from networks.faceid.sphereface import sphere20a

import torchvision
from torchvision import transforms

from loss import AngleLoss

from utils import printoneline, dt, freeze_model, unfreeze_model
from common import CKPT_DIR, LOGS_DIR

def save_model(model, ckpt_path):
    torch.save(model.state_dict(), ckpt_path)

stop_flag = False
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    global stop_flag
    stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)


def global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr):
    optimizer.zero_grad()
    imgs, labels = sample
    imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

    imgs = (imgs*255 - 127.5)/128
    # compute output
    optimizer.zero_grad()
    outputs = faceid(imgs)
    loss = faceid_criterion(outputs, labels)
    loss.backward()
    
#         torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 10)
#         torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
    
    optimizer.step()

    cur_loss = loss.data.cpu().numpy().item()
    train_loss_arr.append(cur_loss)
    total_loss += cur_loss

    outputs = outputs[0] # 0=cos_theta 1=phi_theta
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels.data).sum().cpu().item()

    grads = []
    for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
        grads.append([idx, p.grad.data.norm(2).item()])
    cur_grad_norm = np.sum(grads)

    printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f grad: %.4f' % 
                 (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, cur_loss, 
                 cur_grad_norm))

    return loss, total, correct, total_loss
        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
    for batch_idx, sample in enumerate(dataloader):
        if stop_flag:
            break
        loss, total, correct, total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)

    return total, correct, total_loss

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='name of the experiment')
    parser.add_argument('-d', '--device', type=str, required=True,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    
    train_data_dir = "/tmp/1st_udnet7pa_input/"
#     train_data_dir = "/tmp/CASIA-WebFace-sphereface/"
    transform = transforms.Compose([
#                              transforms.RandomCrop((112, 96)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor()
                         ])
    dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    dataloader_train = torch.utils.data.dataloader.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=14)
    
    fdn = FeatureDenoiser(n_channels = 3)
    faceid = sphere20a(dn_block=fdn)
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/01.01_sphere20a_20.pth" #trained for high noise
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/faceid_joint_3stages_20.01_8"
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "ckpt/1stage_udnet_fixed_sphereface_27.01/faceid/faceid_1stage_udnet_fixed_sphereface_27.01_30"

#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_07.02_fixed/faceid/weights_74"
#     faceid_ckpt_path = "ckpt/1stage_udnet_sphereface_08.02_finetune/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "ckpt/joint_07.02_finetune_08.02/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/sphereface_13.02/faceid/weights_10"
#     faceid_ckpt_path = "/home/safin/ckpt/sphereface_clean/sphere20a_19.pth"
    faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/1st_udnet7pa_sphereface_dn_24.02/faceid/weights_39"
    faceid.load_state_dict(torch.load(faceid_ckpt_path), strict=False)
    faceid = faceid.cuda()
#     freeze_model(faceid)
#     unfreeze_model(faceid.dn_block)
#     unfreeze_model(faceid.fc6)
#     faceid.dn_block.train()
    
    

#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0001)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
#     denoise_criterion = nn.L1Loss().cuda()
    faceid_criterion = AngleLoss().cuda()
    
    cur_logs_path = os.path.join(LOGS_DIR, args.name)
    os.makedirs(cur_logs_path, exist_ok=True)
    
    cur_ckpt_path = os.path.join(CKPT_DIR, args.name)
    os.makedirs(cur_ckpt_path, exist_ok=True)
    faceid_ckpt_path = os.path.join(cur_ckpt_path, "faceid")
    os.makedirs(faceid_ckpt_path, exist_ok=True)
    
    total_train_loss_arr = []
    total_train_acc_arr = []

    lr_milstones = [5, 10, 20, 40]
#     scheduler = MultiStepLR(optimizer, lr_milstones, gamma=0.9)
    lr = 0.01
    for epoch in range(args.epochs):
        total = 0
        correct = 0
        train_loss_arr = []
        total_loss = 0
#         faceid_w = 1
#         denoise_w = 0
#         if epoch >= 1:
#             faceid_w = 1
#         else:
#             faceid_w = 0
#         scheduler.step()
        if epoch in [0,10,15,18,30]:
            if epoch!=0: lr *= 0.5
            optimizer = optim.SGD(faceid.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        total, correct, total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
    
        save_model(faceid, os.path.join(faceid_ckpt_path, "weights_%d" % epoch))
#         torch.save(denoiser.state_dict(), os.path.join(denoiser_ckpt_path, "weights_%d" % epoch))
        
        total_train_loss_arr.append(np.mean(train_loss_arr))
        np.save(os.path.join(cur_logs_path,"train_loss_" + args.name), np.asarray(total_train_loss_arr))
        
        total_train_acc_arr.append(100. * correct/total)
        np.save(os.path.join(cur_logs_path,"train_faceid_acc_" + args.name), np.asarray(total_train_acc_arr))

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
        np.save(os.path.join(cur_logs_path,"train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))
        print("\n")
        
#         total = 0
#         correct = 0
#         train_loss_arr = []
#         total_loss = 0
#         train_epoch(dataloader_val, None, total, correct, total_loss, train_loss_arr)
#         print("\n")
#         torch.save(denoiser.state_dict(), ckpt_path + "denoiser_" + args.name + "_%d" % epoch)
#         np.save("train_loss_" + args.name + "_%d" % epoch, np.asarray(train_loss_arr))

        
        if stop_flag:
            break
    #for l1, l2 in zip(parameters_start,list(model.parameters())):
    #    print(np.array_equal(l1.data.numpy(), l2.data.numpy()))
    print("Done.")