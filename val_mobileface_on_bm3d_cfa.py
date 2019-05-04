#!/usr/bin/python3
import argparse
import os
import tqdm

import torch
import torch.optim as optim
import numpy as np

from networks.faceid.sphereface import sphere20a
from networks.faceid.mobile import MobileFacenet

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


def test_w_denoiser(model, denoiser, test_dataloader, l2dist):
    model.eval()
    denoiser.eval()
    labels_arr = []
    distances_arr = []
    for batch_idx, data in tqdm.tqdm(enumerate(test_dataloader)):
        data_x, data_y, data_label = data
        data_x, data_y, data_label = data_x.cuda(), data_y.cuda(), data_label.cuda()
        
#         data_x = 255*data_x
#         sigma = pydlutil.wmad_estimator(data_x).cuda()
#         denoised_x = denoiser(data_x, sigma)
#         denoised_x = (denoised_x-127.5)/127.5
        
#         data_y = 255*data_y
#         sigma = pydlutil.wmad_estimator(data_y).cuda()
#         denoised_y = denoiser(data_y, sigma)
#         denoised_y = (denoised_y-127.5)/127.5

        data_x = 2*data_x - 1
        data_y = 2*data_y - 1
        out_x = model(data_x)
        out_y = model(data_y)
        
        out_dists = l2dist(out_x, out_y)
        
        distances_arr += out_dists.data.cpu().numpy().tolist()
        labels_arr += data_label.data.cpu().numpy().tolist()
    
    return np.asarray(distances_arr), np.asarray(labels_arr)

from PIL import Image

def test(model, dataloader, l2dist):
    model.eval()
    labels_arr = []
    distances_arr = []
    for batch_idx, data in enumerate(dataloader):
        data_x, data_y, data_label = data
        batch_size = data_x.size(0)
        data_x = M * data_x
        data_y = M * data_y

        data_x = linrgb_to_srgb(bilinear(data_x))
        data_x = data_x/0.6
#         data_x = data_x.clamp(0,1)
        data_y = linrgb_to_srgb(bilinear(data_y))
        data_y = data_y/0.6
#         data_y = data_y.clamp(0,1)

#         Image.fromarray((data_x[0] * 255).permute(1,2,0).numpy().clip(0,255).astype(np.uint8)).save("a.png")
#         Image.fromarray((data_y[0] * 255).permute(1,2,0).numpy().clip(0,255).astype(np.uint8)).save("b.png")
#         break

        data_x = (data_x*255 - 127.5)/128
        data_y = (data_y*255 - 127.5)/128

#         imglist = [data_x.data.cpu().numpy(), data_x.data.cpu().numpy()[:,:,:,::-1], data_y.data.cpu().numpy(), data_y.data.cpu().numpy()[:,:,:,::-1]]

#         img = np.vstack(imglist)
#         img = torch.from_numpy(img).float().cuda()
#         output = model(img)
#         f = output.data
        
#         cur_batch_size = data_x.size(0)
#         out_x, out_y = f[:cur_batch_size], f[2*cur_batch_size:3*cur_batch_size]
#         out_dists = l2dist(out_x, out_y)
        
#         distances_arr += out_dists.data.cpu().numpy().tolist()
#         labels_arr += data_label.data.cpu().numpy().tolist()

        faceid_input = torch.cat([data_x, data_y]).cuda()
        out = model(faceid_input)
        out_dists = l2dist(out[:batch_size], out[batch_size:])
        distances_arr += out_dists.detach().cpu().numpy().tolist()
        labels_arr += data_label.numpy().tolist()
    
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

import torch.nn as nn
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
Module = nn.Module
import collections
from itertools import repeat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validation script')
    parser.add_argument('-d', '--device', type=str, required=True,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    
    im_shape=(112,96)
    M = generate_mask(im_shape).unsqueeze(0)
    
    faceid = MobileFacenet()
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
#     faceid_ckpt_path = "/home/safin/ckpt/sphereface_clean/sphere20a_19.pth"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/1st_udnet7pa_sphereface_dn_24.02/faceid/weights_1"
#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/mobile_04.04/faceid/weights_90"

#     faceid_ckpt_path = "/home/safin/ms-thesis/ckpt/joint_14.04_dnfr/faceid/weights_22"
    faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_16.04/faceid/weights_90"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_on_bm3d_26.04/faceid/weights_30"
#     faceid_ckpt_path = "/home/safin/FaceReID/ckpt/mobile_on_bm3d_30.04/faceid/weights_40"
    model_state = torch.load(faceid_ckpt_path) 
#     module_state = torch.load(faceid_ckpt_path)
#     model_state = OrderedDict()
#     for k, v in module_state.items():
#         model_state[k[7:]] = v
    faceid.load_state_dict(model_state)
    faceid = faceid.cuda()

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

    
    transform = basic_transform #bayer_noised_transform #basic_transform #noise_transform
#     lfw_data_dir = "/home/safin/datasets/lfw/lfw-sphereface/"
    lfw_data_dir = "/home/safin/datasets/lfw/lfw_denoised/"
    lfw_dataset = LFWDataset(lfw_data_dir, "/home/safin/datasets/lfw/pairs.txt", transform, "png")
    dataloader_test = torch.utils.data.dataloader.DataLoader(lfw_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12)
    
    print("The number of parameters:", sum(p.numel() for p in faceid.parameters()))

    l2dist = torch.nn.CosineSimilarity().cuda()
    dists, labels = test(faceid, dataloader_test, l2dist)
    
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
    