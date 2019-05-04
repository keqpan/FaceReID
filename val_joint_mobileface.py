#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.optim as optim
import numpy as np

from networks.faceid.sphereface import sphere20a
from networks.faceid.mobile import MobileFacenet
from utils import printoneline, dt, save_model, load_model

from networks.denoise.pydl import ResNet_Den
from MMNet_TBPTT import *
from problems import *

from transforms.noising import GaussianNoise
from transforms.noising import RawNoiseBayer
import torchvision
from torchvision import transforms

from utils import printoneline, dt, KFold

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
from collections import OrderedDict

from PIL import Image
import numpy as np
idx = 0

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

def one_pass_w_denoiser(data, faceid, denoiser, test_dataloader, l2dist):
    data_x, data_y, data_label = data
    batch_size = data_x.size(0)

    data_x = data_x*255
    data_y = data_y*255
    imgs = torch.cat([data_x, data_y]).cuda(non_blocking=True)
    denoiser_out = denoiser.forward_all_iter(imgs, M, init=False, noise_estimation=True)
    denoiser_out = (denoiser_out - 127.5)/128
    out = faceid(denoiser_out)
    out_dists = l2dist(out[:batch_size], out[batch_size:])

    return out_dists.detach().cpu().numpy().tolist(), data_label.numpy().tolist()
    
def test_w_denoiser(faceid, denoiser, test_dataloader, l2dist):
    faceid.eval()
    denoiser.eval()
    labels_arr = []
    distances_arr = []
    for batch_idx, data in tqdm.tqdm(enumerate(test_dataloader)):
        if stop_flag:
            break

        distances, labels = one_pass_w_denoiser(data, faceid, denoiser, test_dataloader, l2dist)
        distances_arr += distances
        labels_arr += labels
#         denoiser_out = imgs
#         L = estimate_noise(imgs)

#         xpre = 0
#         for i in range(max_iter):
#             denoiser_out, xpre = denoiser.forward(denoiser_out, xpre, imgs, M, L, i)
        
#         denoiser_out_y = denoiser.forward_all_iter(data_y, M, init=False, noise_estimation=True)
#         denoiser_out_x = denoiser_out_x.detach().permute(0,2,3,1).cpu().numpy()
#         denoiser_out_y = denoiser_out_y.detach().permute(0,2,3,1).cpu().numpy()
        
#         Image.fromarray(denoiser_out_x[idx].astype(np.uint8)).save("x.png")
#         Image.fromarray(denoiser_out_y[idx].astype(np.uint8)).save("y.png")
#         break

#         denoiser_out_y = (denoiser_out_y - 127.5)/128
#         data_x = 255*data_x
#         sigma = pydlutil.wmad_estimator(data_x).cuda()
#         denoised_x = denoiser(data_x, sigma)
#         denoised_x = (denoised_x-127.5)/127.5
        
#         data_y = 255*data_y
#         sigma = pydlutil.wmad_estimator(data_y).cuda()
#         denoised_y = denoiser(data_y, sigma)
#         denoised_y = (denoised_y-127.5)/127.5


#         out_y = faceid(denoiser_out_y)

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

def k_fold_eval(dists, labels):
    thresholds = np.arange(-1.0, 1.0, 0.001)
    acc_arr = []
    for pairs in KFold(n=6000, n_folds=10):
        train_pairs, test_pairs = pairs
        t, _ = find_best(thresholds, dists[train_pairs], labels[train_pairs])
        acc_arr.append(eval_acc(t, dists[test_pairs], labels[test_pairs]))
    return np.mean(acc_arr), np.std(acc_arr)

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

# https://github.com/Xiaoccer/MobileFaceNet_Pytorch
# https://github.com/wujiyang/Face_Pytorch   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation script')
    parser.add_argument('-d', '--device', type=str, required=True,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True 
    
    M = generate_mask((112, 96), pattern='RGGB').permute(2,0,1).unsqueeze(0).cuda()
    
    denoiser = ResNet_Den(5, weightnorm=True)
    denoiser = denoiser.cuda()
    
    max_iter = 10
    mmnet = MMNet(denoiser, max_iter=max_iter)
#     mmnet_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_15.04_dnfr/denoiser/weights_15"
#     mmnet_ckpt_path = "/home/safin/ms-thesis/ckpt/resdnet_09.04/denoiser/weights_90"
    mmnet_ckpt_path = "/home/safin/ms-thesis/ckpt/20.04_resdnet_5it/denoiser/weights_200"
    mmnet_ckpt_path = "/home/safin/FaceReID/ckpt/joint_20.04/mmnet/weights_38"
    mmnet = load_model(mmnet, mmnet_ckpt_path, multigpu_mode = False)
    mmnet.eval()

    
    faceid = MobileFacenet()
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_15.04_dnfr/faceid/weights_15"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/mobile_04.04/faceid/weights_90"
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_16.04/faceid/weights_90"
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/joint_20.04/faceid/weights_38"
    model_state = torch.load(faceid_ckpt_path)
#     module_state = torch.load(faceid_ckpt_path)
#     model_state = OrderedDict()
#     for k, v in module_state.items():
#         model_state[k[7:]] = v
    faceid.load_state_dict(model_state)
    faceid = faceid.cuda()
#     faceid = load_model(faceid, faceid_ckpt_path, multigpu_mode = False)
    faceid.eval()
    
    basic_transform = transforms.Compose([
                             transforms.ToTensor()
                         ])
    noise_transform = transforms.Compose([
                             transforms.ToTensor(),
                             GaussianNoise(high_noise_std_arr, clamp=[0,1])
                         ])
    a = 0.15
    b = 0.15
    bayer_noised_transform = transforms.Compose([
                             RawNoiseBayer(a, b, 0.7, 0.6),
                             transforms.ToTensor()
                         ])

    
    transform = bayer_noised_transform #basic_transform #noise_transform
    lfw_data_dir = "/home/safin/datasets/lfw/lfw-sphereface/"
    lfw_dataset = LFWDataset(lfw_data_dir, "/home/safin/datasets/lfw/pairs.txt", transform, "png")
    dataloader_test = torch.utils.data.dataloader.DataLoader(lfw_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    
    print("The number of parameters:", sum(p.numel() for p in faceid.parameters()))

    l2dist = torch.nn.CosineSimilarity().cuda()
#     dists, labels = test(faceid, dataloader_test, l2dist)
    dists, labels = test_w_denoiser(faceid, mmnet, dataloader_test, l2dist)
    
    print("SphereFace, tested on high noised with denoiser:", k_fold_eval(dists, labels))

    
    
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
    