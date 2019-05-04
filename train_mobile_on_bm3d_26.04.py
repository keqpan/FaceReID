#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


import torchvision
from torchvision import transforms

from loss import AngleLoss

from utils import printoneline, dt
from transforms.noising import RawNoise
from datasets.noised import DemosaicDataset
from networks.faceid.mobile import MobileFacenet
from networks.faceid.mobile import ArcMarginProduct
from loss import PSNR

from networks.denoise.pydl import ResNet_Den

from MMNet_TBPTT import *
from problems import *

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler

from common import CKPT_DIR, LOGS_DIR
import itertools
from collections import OrderedDict

def freeze_model(model):
#     model.train(False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
        
def prepare_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

import base
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    base.stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

import numpy as np
import torch
import torch.nn.functional as F
import math, random
from PIL import Image


def bilinear(y):
    r""" Initialize with bilinear interpolation"""
    F_r = torch.FloatTensor([[1,2,1],[2,4,2],[1,2,1]])/4
    F_b = F_r
    F_g = torch.FloatTensor([[0,1,0],[1,4,1],[0,1,0]])/4
    bilinear_filter = torch.stack([F_r,F_g,F_b])[:,None]
    if y.is_cuda:
        bilinear_filter = bilinear_filter.cuda()
    res = F.conv2d(y, bilinear_filter,padding=1, groups=3)
    return res


def linrgb_to_srgb(img):
    """ Convert linRGB color space to sRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
#     assert img.dtype in [np.float32, np.float64]
    if isinstance(img, np.ndarray):
        img = img.copy()
    elif isinstance(img, torch.Tensor):
        img = img.clone()
    mask = img <= 0.0031308
    img[~mask] = (img[~mask]**(1/2.4))*(1.055) - 0.055
    img[mask] = img[mask] * 12.92
    return img

def srgb_to_linrgb(img):
    """ Convert sRGB color space to linRGB 
        https://en.wikipedia.org/wiki/SRGB
    """
#     assert img.dtype in [np.float32, np.float64] 
    if isinstance(img, np.ndarray):
        img = img.copy()
    elif isinstance(img, torch.Tensor):
        img = img.clone()
    mask = img <= 0.04045
    img[~mask] = ((img[~mask]+0.055)/1.055)**2.4
    img[mask] = img[mask] / 12.92
    return img

def generate_mask(im_shape, pattern='RGGB'):
    if pattern == 'RGGB':
        # pattern RGGB
        r_mask = torch.zeros(im_shape)
        r_mask[0::2, 0::2] = 1

        g_mask = torch.zeros(im_shape)
        g_mask[::2, 1::2] = 1
        g_mask[1::2, ::2] = 1

        b_mask = torch.zeros(im_shape)
        b_mask[1::2, 1::2] = 1

        mask = torch.zeros((3,) + im_shape)
        mask[0, :, :] = r_mask
        mask[1, :, :] = g_mask
        mask[2, :, :] = b_mask
        return mask

    
from base import BaseExpRunner

class JointTrainer(BaseExpRunner):
    num_avg_batches = 2 # 64 # 32
    
    def global_forward(self, sample, batch_idx):
        mosaic, groundtruth, labels = sample
        mosaic, groundtruth, labels = mosaic.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        denoised_imgs = linrgb_to_srgb(bilinear(M*mosaic))
        imgs = denoised_imgs/0.6
        imgs = denoised_imgs.clamp(0,255)
        imgs = (imgs-127.5)/128
                  
        if (batch_idx+1)%self.num_avg_batches == 0:
            make_step = True
        else:
            make_step = False

        if batch_idx%self.num_avg_batches == 0:
            zero_grad = True
        else:
            zero_grad = False
            
        if zero_grad:
            optimizer.zero_grad()
        raw_logits = faceid(imgs)
        outputs = arcmargin(raw_logits, labels)
        loss = faceid_criterion(outputs, labels)/self.num_avg_batches
        loss.backward()

        if make_step:
            optimizer.step()

        grads = []
        for idx, par in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
            grads.append(par.grad.data.norm(2).item())
        cur_grad_faceid_norm = np.sum(grads)
        
        grads = []
        for idx, par in enumerate(list(filter(lambda p: p.grad is not None, arcmargin.parameters()))):
            grads.append(par.grad.data.norm(2).item())
        cur_grad_arcmargin_norm = np.sum(grads)
        

        cur_psnr = float(PSNR(denoised_imgs, groundtruth).mean())
    #     psnr_list = calculate_psnr_fast(denoised_imgs/255, groundtruth/255)
    #     mean_psnr = np.array(psnr_list)
    #     mean_psnr = mean_psnr[mean_psnr != np.inf].mean()
        faceid_loss = float(loss)
        cur_loss = faceid_loss
        self.tmp_logs_dict['faceid_loss'].append(faceid_loss)

        self.total_loss += cur_loss # FIXME
        if batch_idx % 50 == 0:
            printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | faceid: %.4f |  gradID: %.4f gradM: %.4f psnr: %.4f' % 
                     (self.cur_epoch, self.total_loss/(batch_idx+1), batch_idx, faceid_loss, cur_grad_faceid_norm, cur_grad_arcmargin_norm, cur_psnr))


sig_read_linspace = np.linspace(-3,-1.5,4)
sig_shot_linspace = np.linspace(-2,-1,4)

sig_read = sig_read_linspace[3]
sig_shot = sig_shot_linspace[2]
a = np.power(10., sig_read)
b = np.power(10., sig_shot)

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
    
    if len(args.device)>1:
        multigpu_mode = True
    else:
        multigpu_mode = False

    im_shape=(112,96)
    M = generate_mask(im_shape).unsqueeze(0).cuda()
#     M = generate_mask((112, 96), pattern='RGGB').permute(2,0,1).unsqueeze(0).cuda()
#     if multigpu_mode:
#         M = M.repeat(args.batch_size,1,1,1)

    
    train_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_train_idxs.npy")
    val_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_test_idxs.npy")
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices[:50])

#     a = 0.15
#     b = 0.15
#     dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)

    train_data_dir = "/home/safin/datasets/CASIA_BM3D_CFA/"
    transform = transforms.Compose([
                             transforms.ToTensor()
                         ])
    noised_dataset = DemosaicDataset(train_data_dir, transform=transform)

    dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=32)
    dataloader_val = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=val_sampler, batch_size=8, pin_memory=True, num_workers=12)
    

    faceid = MobileFacenet() 
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_dnfr_16.04/faceid/weights_30"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_dnfr_22.04/faceid/weights_90"
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_16.04/faceid/weights_60"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_13.04/faceid/weights_6"
#     
#     faceid_state_dict_gpu = torch.load(faceid_ckpt_path, map_location=lambda storage, loc: storage)
#     faceid_state_dict = OrderedDict()
#     for k, v in faceid_state_dict_gpu.items():
#         faceid_state_dict[k[7:]] = v
#     faceid.load_state_dict(faceid_state_dict)
#     faceid = load_model(faceid, faceid_ckpt_path, multigpu_mode)
    arcmargin = ArcMarginProduct(128, len(noised_dataset.classes))
#     arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/joint_dnfr_16.04/arcmargin/weights_30"
    arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_16.04/arcmargin/weights_60"
    
    denoise_criterion = nn.L1Loss().cuda()
    # denoise_criterion = nn.MSELoss().cuda()
    faceid_criterion = nn.CrossEntropyLoss().cuda()


#     optimized_params = itertools.chain(mmnet.parameters(), faceid.parameters(), arcmargin.parameters())
    optimized_params = itertools.chain(faceid.parameters(), arcmargin.parameters())

    lr = 0.001
    optimizer = torch.optim.Adam(optimized_params, lr=lr, amsgrad=True)
    lr_milestones = [3, 9, 15, 25]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.7)

    models_dict = {
                    'faceid': {
                         'model': faceid,
                         'load_ckpt': "/home/safin/FaceReID/ckpt/mobile_16.04/faceid/weights_60"
#                         "/home/safin/FaceReID/ckpt/joint_dnfr_22.04/faceid/weights_23"
                    },
                    'arcmargin': {
                         'model': arcmargin,
                         'load_ckpt': "/home/safin/FaceReID/ckpt/mobile_16.04/arcmargin/weights_60"
#                         "/home/safin/FaceReID/ckpt/joint_dnfr_22.04/argcmargin/weights_23"
                    }
                  }
    schedulers_dict = {'general': scheduler}
    optimizers_dict = {'general': optimizer}
    losses_dict = {'L1': denoise_criterion,
                   'FaceID': faceid_criterion}
    log_names = ['denoiser_loss', "faceid_loss"]
    
    trainer = JointTrainer(args.name, models_dict, schedulers_dict, optimizers_dict, losses_dict, log_names)
    trainer.train(dataloader_train, args.epochs)

#         grads = []
#         for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
#             grads.append([idx, p.grad.data.norm(2).item()])
#         np.save(os.path.join(cur_logs_path,"train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))

    print("Done.")
