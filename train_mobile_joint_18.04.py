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

from utils import printoneline, dt, save_model, load_model
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

stop_flag = False
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    global stop_flag
    stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

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

        mask = torch.zeros(im_shape + (3,))
        mask[:, :, 0] = r_mask
        mask[:, :, 1] = g_mask
        mask[:, :, 2] = b_mask
        
        return mask

num_avg_batches = 32
def global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr):
        
    mosaic, groundtruth, labels = sample
    mosaic, groundtruth, labels = mosaic.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True), labels.cuda(non_blocking=True)

#     make_step = True
#     zero_grad = True
    if (batch_idx+1)%num_avg_batches == 0:
        make_step = True
    else:
        make_step = False

    if batch_idx%num_avg_batches == 0:
        zero_grad = True
    else:
        zero_grad = False

    denoised_imgs, cur_grad_dn_norm, cur_grad_faceid_norm, cur_grad_arcmargin_norm, denoiser_loss, faceid_loss = runner.train(mosaic, M, groundtruth, labels, init=False, noise_estimation=True, zero_grad=zero_grad, make_step=make_step, num_avg_batches=num_avg_batches)

    cur_psnr = float(PSNR(denoised_imgs, groundtruth, PIXEL_MAX=255.).mean())
#     psnr_list = calculate_psnr_fast(denoised_imgs/255, groundtruth/255)
#     mean_psnr = np.array(psnr_list)
#     mean_psnr = mean_psnr[mean_psnr != np.inf].mean()
    cur_loss = denoiser_loss + faceid_loss
    train_loss_arr.append(cur_loss)
    total_loss += cur_loss
    if batch_idx % 50 == 0:
        printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | denoise: %.4f faceid: %.4f | gradDN: %.4f gradID: %.4f gradM: %.4f psnr: %.4f' % 
                 (epoch, total_loss/(batch_idx+1), batch_idx, denoiser_loss, faceid_loss, cur_grad_dn_norm, cur_grad_faceid_norm, cur_grad_arcmargin_norm, cur_psnr))

    return total_loss

        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
    for batch_idx, sample in enumerate(dataloader):
        if stop_flag:
            break
        total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)

    return total_loss

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
    
    M = generate_mask((112, 96), pattern='RGGB').permute(2,0,1).unsqueeze(0).cuda()
    if multigpu_mode:
        M = M.repeat(args.batch_size,1,1,1)

    
    train_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_train_idxs.npy")
    val_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_test_idxs.npy")
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices[:50])
#     train_data_dir = "/tmp/CASIA-WebFace-sphereface"
#     train_data_dir = "/tmp/QMUL_96x112"


#     a = 0.15
#     b = 0.15
#     transform = transforms.Compose([
#                          transforms.RandomCrop((112,96)),
#                          transforms.RandomHorizontalFlip(),
# #                          RawNoise(a, b, 0.7, 0.6),
#                          transforms.ToTensor()
#                      ])
#     dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
#     dataloader_train = torch.utils.data.dataloader.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=14)
    train_data_dir = "/home/safin/datasets/CASIA-WebFace/"
    transform = transforms.Compose([
                             transforms.ToTensor()
                         ])
    noised_dataset = DemosaicDataset(train_data_dir, transform=transform)
#     dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
    dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=32)
#     dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=25)
    dataloader_val = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=val_sampler, batch_size=8, pin_memory=True, num_workers=12)
    
    denoiser = ResNet_Den(5, weightnorm=True)
    n_ckpt = 30
    denoiser_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_dnfr_16.04/denoiser/weights_30"
#     denoiser_ckpt_path = "/home/safin/ckpt/1st_udnetpa/weigths_"+str(n_ckpt)
#     denoiser_ckpt_path = "/home/safin/ms-thesis/ckpt/dncnn_13.03_mse/denoiser/weights_0"
#     denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     freeze_model(denoiser)
    denoiser = denoiser.cuda()
    denoise_criterion = nn.L1Loss().cuda()
       
    max_iter = 5
    mmnet = MMNet(denoiser, max_iter=max_iter)
    mmnet.cuda()
#     mmnet_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_13.04/denoiser/weights_6"
#     mmnet.load_state_dict(torch.load(mmnet_ckpt_path, map_location=lambda storage, loc: storage))
#     mmnet = load_model(mmnet, mmnet_ckpt_path, multigpu_mode)
#     mmnet.eval()
#     freeze_model(mmnet)

    faceid = MobileFacenet() 
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_dnfr_16.04/faceid/weights_30"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_16.04/faceid/weights_60"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_13.04/faceid/weights_6"
#     
#     faceid_state_dict_gpu = torch.load(faceid_ckpt_path, map_location=lambda storage, loc: storage)
#     faceid_state_dict = OrderedDict()
#     for k, v in faceid_state_dict_gpu.items():
#         faceid_state_dict[k[7:]] = v
#     faceid.load_state_dict(faceid_state_dict)
    faceid = load_model(faceid, faceid_ckpt_path, multigpu_mode)
    faceid_criterion = nn.CrossEntropyLoss().cuda()
    

    ArcMargin = ArcMarginProduct(128, len(noised_dataset.classes))
    arcmargin_ckpt_path = "/home/safin/FaceReID/ckpt/joint_dnfr_16.04/arcmargin/weights_30"
    ArcMargin = load_model(ArcMargin, arcmargin_ckpt_path, multigpu_mode)

    optimized_params = itertools.chain(mmnet.parameters(), faceid.parameters(), ArcMargin.parameters())

    lr = 0.001
    optimizer = torch.optim.Adam(optimized_params, lr=lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)

    runner = TBPTT_faceid(mmnet, faceid, denoise_criterion, faceid_criterion, ArcMargin, 5, 5, optimizer, max_iter=max_iter, clip_grad=0.25)
    runner = runner.cuda()
    
#sphere20a(classnum=len(dataset_train.classes))
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
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/sr_faceid_27.03/faceid/weights_13"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/sr_qmul_29.03/faceid/weights_12"
#     model_state = torch.load(faceid_ckpt_path)
#     del model_state['fc6.weight']
#     faceid.load_state_dict(model_state, strict=False)


#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0001)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
#     denoise_criterion = nn.L1Loss().cuda()



    
    cur_logs_path = os.path.join(LOGS_DIR, args.name)
    os.makedirs(cur_logs_path, exist_ok=True)
    
    cur_ckpt_path = os.path.join(CKPT_DIR, args.name)
    os.makedirs(cur_ckpt_path, exist_ok=True)
    denoiser_ckpt_path = os.path.join(cur_ckpt_path, "denoiser")
    os.makedirs(denoiser_ckpt_path, exist_ok=True)
    faceid_ckpt_path = os.path.join(cur_ckpt_path, "faceid")
    os.makedirs(faceid_ckpt_path, exist_ok=True)
    arcmargin_ckpt_path = os.path.join(cur_ckpt_path, "arcmargin")
    os.makedirs(arcmargin_ckpt_path, exist_ok=True)
    
    total_train_loss_arr = []
    total_train_acc_arr = []

    lr_milstones = [5, 10, 20, 40]
#     scheduler = MultiStepLR(optimizer, lr_milstones, gamma=0.9)

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
#         
#         if epoch in [0,10,15,18]:
#             if epoch!=0: lr *= 0.1 #lr *= 0.9
#             optimizer = optim.SGD(itertools.chain(faceid.parameters(), ArcMargin.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler.step()
        total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
        
        save_model(mmnet, os.path.join(denoiser_ckpt_path, "weights_%d" % epoch), multigpu_mode)
        save_model(faceid, os.path.join(faceid_ckpt_path, "weights_%d" % epoch), multigpu_mode)
        save_model(ArcMargin, os.path.join(arcmargin_ckpt_path, "weights_%d" % epoch), multigpu_mode)
        
        total_train_loss_arr.append(np.mean(train_loss_arr))
        np.save(os.path.join(cur_logs_path,"train_loss_" + args.name), np.asarray(total_train_loss_arr))
        
#         total_train_acc_arr.append(100. * correct/total)
#         np.save(os.path.join(cur_logs_path,"train_faceid_acc_" + args.name), np.asarray(total_train_acc_arr))

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
