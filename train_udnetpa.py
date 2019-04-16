#!/usr/bin/python3
import argparse
import os
import tqdm

from networks.denoise.pydl import ResDNet
from networks.denoise.pydl import UDNet
from networks.denoise.pydl import UDNetPA

from utils import printoneline, dt, freeze_model, unfreeze_model

import sys
sys.path.append("/home/safin/")
from pydl import utils as pydlutil

from common import CKPT_DIR, LOGS_DIR

from torchvision import datasets, transforms
import random
import torchvision.transforms.functional as TF
from PIL import Image    

import torch
import torch.optim as optim
import torch.nn as nn 
import numpy as np
low_noise_std_arr = (np.arange(5, 25, 4)/255).tolist()
high_noise_std_arr = (np.arange(30, 55, 4)/255).tolist()
transform = transforms.Compose([
#                          transforms.RandomCrop((112,96)),
#                          transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()
                     ])

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
        
from datasets.noised import NoisedDataset
from transforms.noising import GaussianNoise

        
def global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr):
    noised, groundtruth, _ = sample
    noised, groundtruth = noised.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True)
    sigma = pydlutil.wmad_estimator(noised)
    sigma = sigma.cuda(non_blocking=True)

#     faceid_input = denoised/127.5 - 1
# #     faceid_input = groundtruth/127.5 - 1
#     faceid_outputs = faceid(faceid_input)

#     faceid_loss = faceid_criterion(faceid_outputs, labels)
# #     denoiser_loss = denoise_criterion(denoised, groundtruth)
# #             loss = faceid_w*faceid_loss + denoise_w*denoiser_loss
#     loss = faceid_loss# + denoiser_loss
#     cur_loss = loss.data.cpu().numpy().item()
#     train_loss_arr.append(cur_loss)
#     total_loss += cur_loss
    
#     faceid_outputs = faceid_outputs[0] # 0=cos_theta 1=phi_theta
#     _, predicted = torch.max(faceid_outputs.data, 1)
#     total += labels.size(0)
#     correct += predicted.eq(labels.data).sum().cpu().item()
    
    if optimizer is not None:
        optimizer.zero_grad()
        
        denoised = denoiser(noised, sigma)
        denoiser_loss = denoise_criterion(denoised, groundtruth)
        loss = denoiser_loss 

        cur_loss = loss.data.cpu().numpy().item()
        total_loss += cur_loss
        loss.backward()

#         torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 50)
#         torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 500)

        optimizer.step()
        train_loss_arr.append(cur_loss)
        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, denoiser.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])
    #                 print("denoise_grad_norm:", grads)
        cur_grad_norm_dn = np.sum(grads)

#         for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
#             grads.append([idx, p.grad.data.norm(2).item()])
#         cur_grad_norm = np.sum(grads)

        printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | L1: %.4f gradDN: %.4f' % 
                     (epoch, total_loss/(batch_idx+1), batch_idx, denoiser_loss.data.cpu().numpy().item(), 
                     cur_grad_norm_dn)) #denoiser_loss.data.cpu().numpy().item()
    else:
#         printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f' % (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, faceid_loss.data.cpu().numpy().item()))
        cur_grad_norm_dn = 0
        denoised = denoiser(noised, sigma)
        denoiser_loss = denoise_criterion(denoised, groundtruth)
        loss = denoiser_loss 

        total_loss += loss.data.cpu().numpy().item()
        printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | L1: %.4f gradDN: %.4f' % 
                     (epoch, total_loss/(batch_idx+1), batch_idx, denoiser_loss.data.cpu().numpy().item(), 
                     cur_grad_norm_dn))
    
    return loss, total, correct, total_loss
        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
#     global stop_flag
#     i = 0
    for batch_idx, sample in enumerate(dataloader):
#         if i > 40: stop_flag = True
        if stop_flag:
            break
        loss, total, correct, total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)
        
#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
            
#             torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 10)
#             torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
            
#             optimizer.step()
#         i += 1
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
    
    train_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_train_idxs.npy")[:5000]
    val_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_test_idxs.npy")[:1000]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_data_dir = "/tmp/CASIA-WebFace-sphereface/"
    noised_dataset = NoisedDataset(train_data_dir, transform=transform, noise_transform=GaussianNoise(std=high_noise_std_arr, threshold=0.7)) #"/tmp/casia_denoised_joint")
    dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    dataloader_val = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=val_sampler, batch_size=8, pin_memory=True, num_workers=12)
    
#     exp = ExpRunner()
#     exp.init_model(args.device, last_ckpt=args.resume)
#     exp.run_experiments(args.name, args.epochs, batch_size=args.batch_size)
    
    denoiser = UDNetPA(kernel_size = (5, 5),
                input_channels = 3,
                output_features = 32,
                rpa_depth = 9,
                shortcut=(False,True))
    
#     denoiser = UDNet(kernel_size = (5, 5),
#                   input_channels = 3,
#                   output_features = 74,
#                   rbf_mixtures = 51,
#                   rbf_precision = 4,
#                   stages = 1)
    
#     denoiser = ResDNet(kernel_size = (5, 5),
#                     input_channels = 3,
#                     output_features = 32,
#                     rpa_depth = 7,
#                     shortcut=(False,True))
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

#     denoiser_ckpt_path = "/home/safin/FaceReID/ckpt/joint_udnetpa_fixed_16.02_finetune/denoiser/weights_"+str(n_ckpt)
#     denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     freeze_model(denoiser)
    denoiser = denoiser.cuda()
#     denoiser_ckpt_path = "/home/safin/pydl/networks/UDNet/ckpt/model_udnet_14.01_3"

#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0005)
#     optimizer = optim.Adam(faceid.parameters(), lr=0.001)

#     criterion = nn.MSELoss().cuda()
    denoise_criterion = nn.L1Loss().cuda()
    faceid_criterion = AngleLoss().cuda()
    
    cur_logs_path = os.path.join(LOGS_DIR, args.name)
    os.makedirs(cur_logs_path, exist_ok=True)

    cur_ckpt_path = os.path.join(CKPT_DIR, args.name)
    os.makedirs(cur_ckpt_path, exist_ok=True)

    denoiser_ckpt_path = os.path.join(cur_ckpt_path, "denoiser")
    os.makedirs(denoiser_ckpt_path, exist_ok=True)
    
    total_train_loss_arr = []
    total_train_acc_arr = []
    
    lr = 0.0001
    optimizer = optim.Adam(denoiser.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-04)
    lr_milstones = [5, 10, 40]
    scheduler = MultiStepLR(optimizer, lr_milstones, gamma=0.9)

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
        scheduler.step()
#         if epoch in [0,5,10,15,18]:
#             if epoch!=0: lr *= 0.5
#             optimizer = optim.SGD(denoiser.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
        total, correct, total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
        torch.save(denoiser.state_dict(), os.path.join(denoiser_ckpt_path, "weights_%d" % epoch))
        
        total_train_loss_arr.append(np.mean(train_loss_arr))
        np.save(os.path.join(cur_logs_path, "train_loss_" + args.name), np.asarray(total_train_loss_arr))
        
#         total_train_acc_arr.append(100. * correct/total)
#         np.save(os.path.join(cur_logs_path, "train_faceid_acc_" + args.name), np.asarray(total_train_acc_arr))

        grads = []
        for idx, p in enumerate(list(filter(lambda p: p.grad is not None, denoiser.parameters()))):
            grads.append([idx, p.grad.data.norm(2).item()])

#         for idx, p in enumerate(list(filter(lambda p: p.grad is not None, faceid.parameters()))):
#             grads.append([idx, p.grad.data.norm(2).item()])
        np.save(os.path.join(cur_logs_path, "train_grads_" + args.name  + "_%d" % epoch), np.asarray(grads))
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
