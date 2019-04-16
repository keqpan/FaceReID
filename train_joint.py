#!/usr/bin/python3
import argparse
import os
import tqdm

from networks.denoise.pydl import ResDNet
from networks.denoise.pydl import UDNet
from networks.denoise.pydl import UDNetPA

from networks.faceid.sphereface import sphere20a

from utils import printoneline, dt, freeze_model, unfreeze_model

import sys
sys.path.append("/home/safin/")
from pydl import utils as pydlutil

from datasets.noised import NoisedDataset
from transforms.noising import GaussianNoise

from common import CKPT_DIR, LOGS_DIR

from torchvision import datasets, transforms
import random
import torchvision.transforms.functional as TF
from PIL import Image    

import torch
import torch.optim as optim
import torch.nn as nn 
import numpy as np

import itertools

from loss import AngleLoss

low_noise_std_arr = (np.arange(5, 25, 4)/255).tolist()
high_noise_std_arr = (np.arange(30, 55, 4)/255).tolist()

stop_flag = False
def handler(signum, frame):
    print("Shutting down at " + dt() + " ...")
    global stop_flag
    stop_flag = True

import signal
signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler

def global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr):
    optimizer.zero_grad()
    noised, groundtruth, labels = sample
    noised, groundtruth, labels = noised.cuda(non_blocking=True), groundtruth.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    
    sigma = pydlutil.wmad_estimator(noised)
    sigma = sigma.cuda(non_blocking=True)
    denoised_imgs = denoiser(noised, sigma)
    faceid_inputs = (denoised_imgs - 127.5)/128
    # compute output
    optimizer.zero_grad()
    outputs = faceid(faceid_inputs)
    loss = faceid_criterion(outputs, labels)

    
    torch.nn.utils.clip_grad.clip_grad_norm_(faceid.parameters(), 5)
#         torch.nn.utils.clip_grad.clip_grad_norm_(denoiser.parameters(), 10)
    loss.backward()
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
        grads.append(p.grad.data.norm(2).item())
    cur_grad_faceid_norm = np.sum(grads)
    
    grads = []
    for idx, p in enumerate(list(filter(lambda p: p.grad is not None, denoiser.parameters()))):
        grads.append(p.grad.data.norm(2).item())
    cur_grad_dn_norm = np.sum(grads)

    printoneline(dt(),'Te=%d TLoss=%.4f batch=%d | acc: %.4f%% faceid: %.4f gradDN: %.4f grad: %.4f' % 
                 (epoch, total_loss/(batch_idx+1), batch_idx, 100. * correct/total, cur_loss, 
                 cur_grad_dn_norm, cur_grad_faceid_norm))

    return loss, total, correct, total_loss
        
def train_epoch(dataloader, optimizer, total, correct, total_loss, train_loss_arr):
    for batch_idx, sample in enumerate(dataloader):
        if stop_flag:
            break
        loss, total, correct, total_loss = global_forward(sample, batch_idx, optimizer, total, correct, total_loss, train_loss_arr)

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
    
    train_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_train_idxs.npy")
    val_indices = np.load("/home/safin/datasets/CASIA-WebFace/casia_test_idxs.npy")
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_data_dir = "/tmp/CASIA-WebFace-sphereface/"
    transform = transforms.Compose([
                             transforms.RandomCrop((112,96)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor()
                         ])
    noised_dataset = NoisedDataset(train_data_dir, transform=transform, noise_transform=GaussianNoise(std=high_noise_std_arr, threshold=0.7))
#     dataset_train = torchvision.datasets.ImageFolder(train_data_dir, transform=transform)
#     dataloader_train = torch.utils.data.dataloader.DataLoader(dataset_train, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=14)
    dataloader_train = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    dataloader_val = torch.utils.data.dataloader.DataLoader(noised_dataset, sampler=val_sampler, batch_size=8, pin_memory=True, num_workers=12)
    
        
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
    
    denoiser = UDNetPA(kernel_size = (5, 5),
                input_channels = 3,
                output_features = 32,
                rpa_depth = 7,
                shortcut=(False,True))
    n_ckpt = 30
    denoiser_ckpt_path = "/home/safin/ckpt/1st_udnetpa/weigths_"+str(n_ckpt)
    denoiser.load_state_dict(torch.load(denoiser_ckpt_path))
#     freeze_model(denoiser)
    denoiser = denoiser.cuda()
    
    faceid = sphere20a()
    faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/sphereface_on_1st_udnet7pa_27.02/faceid/weights_15"
    faceid.load_state_dict(torch.load(faceid_ckpt_path))
    faceid = faceid.cuda()
    freeze_model(faceid)
#     optimizer = optim.Adam(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=0.0001)

    denoise_criterion = nn.L1Loss().cuda()
    faceid_criterion = AngleLoss().cuda()
    
    cur_logs_path = os.path.join(LOGS_DIR, args.name)
    os.makedirs(cur_logs_path, exist_ok=True)
    
    cur_ckpt_path = os.path.join(CKPT_DIR, args.name)
    os.makedirs(cur_ckpt_path, exist_ok=True)
    faceid_ckpt_path = os.path.join(cur_ckpt_path, "faceid")
    os.makedirs(faceid_ckpt_path, exist_ok=True)

    denoiser_ckpt_path = os.path.join(cur_ckpt_path, "denoiser")
    os.makedirs(denoiser_ckpt_path, exist_ok=True)
    
    total_train_loss_arr = []
    total_train_acc_arr = []

    lr_milstones = [5, 10, 20, 40]
#     scheduler = MultiStepLR(optimizer, lr_milstones, gamma=0.9)
    lr = 0.005 #
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
        if epoch in [0,10,15,18]:
            if epoch!=0: lr *= 0.5 #lr *= 0.9
            optimizer = optim.SGD(itertools.chain(denoiser.parameters(), faceid.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
        total, correct, total_loss = train_epoch(dataloader_train, optimizer, total, correct, total_loss, train_loss_arr)
    
        torch.save(faceid.state_dict(), os.path.join(faceid_ckpt_path, "weights_%d" % epoch))
        torch.save(denoiser.state_dict(), os.path.join(denoiser_ckpt_path, "weights_%d" % epoch))
        
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