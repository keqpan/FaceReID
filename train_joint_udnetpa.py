#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.nn.utils import weight_norm
from torchvision import datasets, transforms


import sys
sys.path.append("/home/safin/")

import math

from pydl.nnLayers import modules
from pydl.nnLayers import init
from pydl.utils import loadmat

from pydl.nnLayers import modules
from pydl.nnLayers import init
from pydl.utils import loadmat
from pydl.nnLayers.cascades import nconv2D, nconv_transpose2D
from pydl.nnLayers.functional.functional import L2Proj
from pydl.utils import formatInput2Tuple, getPad2RetainShape

from pydl.nnLayers.functional import functional as F
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
#            Kc = th.Tensor(kernel_size).add(1).div(2).floor()
#            pad = (int(Kc[0])-1, kernel_size[0]-int(Kc[0]),\
#                   int(Kc[1])-1,kernel_size[1]-int(Kc[1]))              
        
        self.pad = formatInput2Tuple(pad,int,4)
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
        self.resPA = nn.ModuleList([modules.ResidualPreActivationLayer(\
                        rpa_kernel_size1,rpa_kernel_size2,output_features,\
                        rpa_output_features,rpa_bias1,rpa_bias2,1,1,\
                        numparams_prelu1,numparams_prelu2,prelu_init,padType,\
                        rpa_scale1,rpa_scale2,rpa_normalizedWeights,\
                        rpa_zeroMeanWeights,rpa_init,self.shortcut[i]) \
                        for i in range(self.rpa_depth)]) 
        
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
        for m in self.resPA:
            output = m(output)
        
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



from pydl import utils as pydlutil
from torchvision import datasets, transforms
import random
import torchvision.transforms.functional as TF
from PIL import Image

class NoisedDataset(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dirs, transform=None, noise_transform=None, storage_dir=None):

        super(NoisedDataset, self).__init__(dirs, transform)
        self.noise_transform = noise_transform
        self.storage_dir = storage_dir

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''
        img_path, class_id = self.imgs[index]
        img1 = self.loader(img_path)
        
        use_current_img = True
        if self.storage_dir is not None:
            img_path_storage = os.path.join(self.storage_dir, os.path.relpath(img_path, self.root))
            
            if os.path.exists(img_path_storage):
                img2 = self.loader(img_path_storage)
                use_current_img = False

        if use_current_img:
            img2 = self.noise_transform(np.asarray(img1))
            img2 = Image.fromarray(img2.clip(0, 255).astype(np.uint8))
       
        if self.storage_dir is not None and not os.path.exists(img_path_storage):
            os.makedirs(os.path.dirname(img_path_storage), exist_ok=True)
            img2.save(img_path_storage)

        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=(112,96))
        img1 = TF.crop(img1, i, j, h, w)
        img2 = TF.crop(img2, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        
        img1 = 255*img1
        img2 = 255*img2
        sigma = pydlutil.wmad_estimator(img2)

        return img2, img1, class_id, sigma


class GaussianNoise(object):
    def __init__(self, std, mean=0, threshold = 0.5):
        self.std = std
        self.mean = mean
        self.threshold = threshold

    def __call__(self, img):
        if random.random() > self.threshold:
            return img
        
        if isinstance(self.std, list):
            std = np.random.choice(self.std)
        else:
            std = self.std
#         return torch.clamp(img + torch.randn(*img.shape)*std + self.mean, 0, 1)
#         return img + torch.randn(*img.shape)*std + self.mean
        return img + np.random.randn(*img.shape)*std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

low_noise_std_arr = (np.arange(5, 25, 4)).tolist()
high_noise_std_arr = (np.arange(30, 55, 4)).tolist()
transform = transforms.Compose([
#                          transforms.RandomCrop((112,96)),
#                          transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()
                     ])

    
import datetime
import sys

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')
   
ckpt_path = "ckpt/"
train_data_dir = "/tmp/CASIA-WebFace-sphereface/" 

def save_model():
    torch.save(model.state_dict(), ckpt_path + "model_" + args.name + "_%d" % epoch)

stop_flag = False
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    global stop_flag
    stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

from networks import sphere20a
from loss import AngleLoss
import itertools

def freeze_model(model):
#     model.train(False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def prepare_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
logs_dir = "logs"
ckpt_dir = "ckpt"
    
import threading
import queue
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image
def do_work(item):
    img, path = item
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((img*255).clip(0,255).astype(np.uint8)).save(path)

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        do_work(item)
        q.task_done()
        
def global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr):
    noised, groundtruth, labels, sigma = sample
    noised, groundtruth, labels, sigma = noised.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True), labels.cuda(non_blocking=True), sigma.cuda(non_blocking=True)

    denoised = denoiser(noised, sigma)
    faceid_input = denoised/127.5 - 1
#     faceid_input = groundtruth/127.5 - 1
    faceid_outputs = faceid(faceid_input)

    faceid_loss = faceid_criterion(faceid_outputs, labels)
#     denoiser_loss = denoise_criterion(denoised, groundtruth)
#             loss = faceid_w*faceid_loss + denoise_w*denoiser_loss
    loss = faceid_loss# + denoiser_loss
    cur_loss = loss.data.cpu().numpy().item()
    train_loss_arr.append(cur_loss)
    total_loss += cur_loss
    
    faceid_outputs = faceid_outputs[0] # 0=cos_theta 1=phi_theta
    _, predicted = torch.max(faceid_outputs.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels.data).sum().cpu().item()
    
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 50)
        torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 500)

        optimizer.step()

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, denoiser.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
    #                 print("denoise_grad_norm:", grads)
        cur_grad_norm_dn = np.sum(grads)

        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
        cur_grad_norm = np.sum(grads)

        printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f L1: %.4f gradDN: %.4f grad: %.4f' % 
                     (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, faceid_loss.data.cpu().numpy().item(), 
                     0, cur_grad_norm_dn, cur_grad_norm)) #denoiser_loss.data.cpu().numpy().item()
    else:
        printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f' % (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, faceid_loss.data.cpu().numpy().item()))
    
    return loss, total, correct, total_loss
        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
    for batch_idx, sample in enumerate(dataloader):
        if stop_flag:
            break
        loss, total, correct, total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)
        
#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
            
#             torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 10)
#             torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
            
#             optimizer.step()
    return total, correct, total_loss
    
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
    
    train_indices = np.load("/home/safin/casia_train_idxs.npy")
    val_indices = np.load("/home/safin/casia_test_idxs.npy")
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    noised_dataset = NoisedDataset(train_data_dir, transform=transform, noise_transform=GaussianNoise(std=high_noise_std_arr, threshold=0.7), storage_dir=None)#"/tmp/casia_denoised_joint")
    dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=12)
    dataloader_val = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=val_sampler, batch_size=8, pin_memory=True, num_workers=12)
    
#     exp = ExpRunner()
#     exp.init_model(args.device, last_ckpt=args.resume)
#     exp.run_experiments(args.name, args.epochs, batch_size=args.batch_size)
    
    denoiser = UDNetPA(kernel_size = (5, 5),
                input_channels = 3,
                output_features = 32,
                rpa_depth = 7,
                shortcut=(False,True))
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/denoiser_joint_19.01_11"
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/model_udnet_3stages_17.01_4" #trained for high noise
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/denoiser_joint_3stages_20.01_8" 
#     denoiser_ckpt_path =  "/home/safin/FaceReID/ckpt/model_udnet_3stages_17.01_4"
#     denoiser_ckpt_path =  "/home/safin/FaceReID/ckpt/udnet_1stage_20.01/model_udnet_1stage_20.01_5"

#     denoiser_ckpt_path =  "/home/safin/FaceReID/ckpt/1stage_udnet_02.02/model/1stage_udnet_02.02_10"
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/joint_07.02_fixed/denoiser/weights_74"
    n_ckpt = 30
#     denoiser_ckpt_path = "ckpt/1stage_udnet_sphereface_08.02_finetune/denoiser/weights_"+str(n_ckpt)
#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/1st_udnet7pa_14.02/model/weigths_29"
    denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/joint_udnetpa_fixed_16.02_finetune/denoiser/weights_"+str(n_ckpt)
    denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     freeze_model(denoiser)
    denoiser = denoiser.cuda()
#     denoiser_ckpt_path = "/home/safin/pydl/networks/UDNet/ckpt/model_udnet_14.01_3"
#     denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     denoiser = denoiser.cuda()
    
    faceid = sphere20a()
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/01.01_sphere20a_20.pth" #trained for high noise
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/faceid_joint_3stages_20.01_8"
#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "ckpt/1stage_udnet_fixed_sphereface_27.01/faceid/faceid_1stage_udnet_fixed_sphereface_27.01_30"

#     faceid_ckpt_path = "/home/safin/sphereface_pytorch/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_07.02_fixed/faceid/weights_74"
#     faceid_ckpt_path = "ckpt/1stage_udnet_sphereface_08.02_finetune/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "ckpt/joint_07.02_finetune_08.02/faceid/weights_"+str(n_ckpt)
#     faceid_ckpt_path = "ckpt/joint_udnetpa_fixed_14.02/faceid/weights_2"
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_udnetpa_fixed_16.02_finetune/faceid/weights_"+str(n_ckpt)
    faceid.load_state_dict(torch.load(faceid_ckpt_path))
    faceid = faceid.cuda()

#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0005)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
    denoise_criterion = nn.L1Loss().cuda()
    faceid_criterion = AngleLoss().cuda()
    
    cur_logs_path = os.path.join(logs_dir, args.name)
    prepare_path(cur_logs_path)
    
    cur_ckpt_path = os.path.join(ckpt_dir, args.name)
    prepare_path(cur_ckpt_path)
    faceid_ckpt_path = os.path.join(cur_ckpt_path, "faceid")
    prepare_path(faceid_ckpt_path)
    denoiser_ckpt_path = os.path.join(cur_ckpt_path, "denoiser")
    prepare_path(denoiser_ckpt_path)
    
    total_train_loss_arr = []
    total_train_acc_arr = []

#     lr_milstones = [2, 5, 10, 40]
#     scheduler = MultiStepLR(optimizer, lr_milstones, gamma=0.9)
    lr = 0.002
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
        if epoch in [0,5,10,15,18]:
            if epoch!=0: lr *= 0.6
            optimizer = optim.SGD(faceid.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
        total, correct, total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
    
        torch.save(faceid.state_dict(), os.path.join(faceid_ckpt_path, "weights_%d" % epoch))
        torch.save(denoiser.state_dict(), os.path.join(denoiser_ckpt_path, "weights_%d" % epoch))
        
        total_train_loss_arr.append(np.mean(train_loss_arr))
        np.save(os.path.join(cur_logs_path,"train_loss_" + args.name), np.asarray(total_train_loss_arr))
        
        total_train_acc_arr.append(100. * correct/total)
        np.save(os.path.join(cur_logs_path,"train_faceid_acc_" + args.name), np.asarray(total_train_acc_arr))

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, denoiser.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])

        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
        np.save(os.path.join(cur_logs_path,"train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))
        print("\n")
        
        total = 0
        correct = 0
        train_loss_arr = []
        total_loss = 0
        train_epoch(dataloader_val, None, total, correct, total_loss, train_loss_arr)
        print("\n")
#         torch.save(denoiser.state_dict(), ckpt_path + "denoiser_" + args.name + "_%d" % epoch)
#         np.save("train_loss_" + args.name + "_%d" % epoch, np.asarray(train_loss_arr))

        
        if stop_flag:
            break
    #for l1, l2 in zip(parameters_start,list(model.parameters())):
    #    print(np.array_equal(l1.data.numpy(), l2.data.numpy()))
    print("Done.")
